package embedding

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type tokenBatch struct {
	start         int
	end           int
	inputIDs      []int64
	attentionMask []int64
	tokenTypeIDs  []int64
}

type pipelineItem[T any] struct {
	value T
}

type oneAheadPipeline[T any] struct {
	cancel context.CancelFunc
	items  <-chan pipelineItem[T]
	done   <-chan error
	once   sync.Once
	err    error
}

// startOneAheadPipeline runs exactly one producer. The unbuffered handoff lets
// the producer prepare item N+1 while the consumer processes item N, but it
// cannot start N+2 until the consumer accepts N+1.
func startOneAheadPipeline[T any](ctx context.Context, count int, produce func(context.Context, int) (T, error)) *oneAheadPipeline[T] {
	pipelineCtx, cancel := context.WithCancel(ctx)
	items := make(chan pipelineItem[T])
	done := make(chan error, 1)
	go func() {
		defer close(items)
		for i := 0; i < count; i++ {
			value, err := produce(pipelineCtx, i)
			if err != nil {
				done <- err
				return
			}
			select {
			case items <- pipelineItem[T]{value: value}:
			case <-pipelineCtx.Done():
				done <- pipelineCtx.Err()
				return
			}
		}
		done <- nil
	}()
	return &oneAheadPipeline[T]{cancel: cancel, items: items, done: done}
}

// stop cancels production and joins the producer. It is safe to call from both
// an early-return defer and the successful completion path.
func (p *oneAheadPipeline[T]) stop() error {
	p.once.Do(func() {
		p.cancel()
		p.err = <-p.done
	})
	return p.err
}

func (e *Embedder) tokenizeDocuments(texts []string) (inputIDs, attentionMask, tokenTypeIDs []int64, lengths []int, err error) {
	tokenStarted := time.Now()
	inputIDs, attentionMask, tokenTypeIDs, lengths, err = e.tokenizer.EncodeBatchWithLengths(texts, e.maxSeqLen)
	e.telemetry.observe(StageTokenization, time.Since(tokenStarted))
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("embedding documents: tokenizer encode: %w", err)
	}
	e.telemetry.observeTokens(attentionMask, len(texts), e.maxSeqLen)
	return inputIDs, attentionMask, tokenTypeIDs, lengths, nil
}

// embedFixedPipeline overlaps tokenization of chunk N+1 with inference of N.
// The caller holds e.mu, so this producer is the only tokenizer user and the
// consumer is the only ONNX session user for the lifetime of the pipeline.
func (e *Embedder) embedFixedPipeline(ctx context.Context, texts []string) ([][]float32, error) {
	chunkCount := (len(texts) + e.batchSize - 1) / e.batchSize
	pipeline := startOneAheadPipeline(ctx, chunkCount, func(producerCtx context.Context, chunk int) (tokenBatch, error) {
		if err := producerCtx.Err(); err != nil {
			return tokenBatch{}, err
		}
		start := chunk * e.batchSize
		end := min(start+e.batchSize, len(texts))
		ids, mask, types, _, err := e.tokenizeDocuments(texts[start:end])
		if err != nil {
			return tokenBatch{}, fmt.Errorf("tokenizing batch chunk [%d:%d] of %d documents: %w", start, end, len(texts), err)
		}
		return tokenBatch{start: start, end: end, inputIDs: ids, attentionMask: mask, tokenTypeIDs: types}, nil
	})
	defer func() { _ = pipeline.stop() }()

	e.logger.Debug("running pipelined batch inference (persistent batch session)",
		"texts", len(texts), "chunkSize", e.batchSize, "seqLen", e.maxSeqLen)
	results := make([][]float32, 0, len(texts))
	for item := range pipeline.items {
		batch := item.value
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var (
			vecs [][]float32
			err  error
		)
		e.onORTThread(func() {
			vecs, err = e.batchSess.runBatch(batch.end-batch.start,
				batch.inputIDs, batch.attentionMask, batch.tokenTypeIDs)
		})
		if err != nil {
			return nil, fmt.Errorf("embedding batch chunk [%d:%d] of %d documents: %w",
				batch.start, batch.end, len(texts), err)
		}
		results = append(results, vecs...)
	}
	if err := pipeline.stop(); err != nil {
		return nil, err
	}
	return results, nil
}
