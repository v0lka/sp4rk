package embedding

import (
	"context"
	"fmt"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// dynamicBenchSession evaluates ONNX Runtime's dynamic-shape API separately
// from the production fixed-shape bucket cache. It is benchmark-only by design.
type dynamicBenchSession struct {
	session   *ort.DynamicAdvancedSession
	hiddenDim int
	telemetry *Telemetry
}

func newDynamicBenchSession(modelPath string, hiddenDim int, opts *ort.SessionOptions, telemetry *Telemetry) (*dynamicBenchSession, error) {
	started := time.Now()
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("creating dynamic ONNX session: %w", err)
	}
	telemetry.observeSession(time.Since(started))
	return &dynamicBenchSession{session: session, hiddenDim: hiddenDim, telemetry: telemetry}, nil
}

func (s *dynamicBenchSession) destroy() {
	if s.session != nil {
		_ = s.session.Destroy()
		s.session = nil
	}
}

func (s *dynamicBenchSession) run(inputIDs, mask, types []int64, batchSize, seqLen int) ([][]float32, error) {
	shape := ort.NewShape(int64(batchSize), int64(seqLen))
	idsTensor, err := ort.NewTensor(shape, inputIDs)
	if err != nil {
		return nil, err
	}
	defer func() { _ = idsTensor.Destroy() }()
	maskTensor, err := ort.NewTensor(shape, mask)
	if err != nil {
		return nil, err
	}
	defer func() { _ = maskTensor.Destroy() }()
	typeTensor, err := ort.NewTensor(shape, types)
	if err != nil {
		return nil, err
	}
	defer func() { _ = typeTensor.Destroy() }()

	outputShape := ort.NewShape(int64(batchSize), int64(seqLen), int64(s.hiddenDim))
	output, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}
	defer func() { _ = output.Destroy() }()

	started := time.Now()
	if err := s.session.Run(
		[]ort.Value{idsTensor, maskTensor, typeTensor},
		[]ort.Value{output},
	); err != nil {
		return nil, err
	}
	s.telemetry.observeInference(batchSize, batchSize, time.Since(started))
	return meanPoolAndNormalize(output.GetData(), mask, batchSize, seqLen, s.hiddenDim), nil
}

func benchmarkDynamicDocuments(ctx context.Context, tok *Tokenizer, session *dynamicBenchSession, texts []string, maxSeqLen int) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	started := time.Now()
	ids, mask, types, lengths, err := tok.EncodeBatchWithLengths(texts, maxSeqLen)
	session.telemetry.observe(StageTokenization, time.Since(started))
	if err != nil {
		return nil, err
	}
	seqLen := 2
	for _, length := range lengths {
		seqLen = max(seqLen, length)
	}
	batchSize := len(texts)
	compactIDs := make([]int64, batchSize*seqLen)
	compactMask := make([]int64, batchSize*seqLen)
	compactTypes := make([]int64, batchSize*seqLen)
	for row := range batchSize {
		src, dst := row*maxSeqLen, row*seqLen
		copy(compactIDs[dst:dst+seqLen], ids[src:src+seqLen])
		copy(compactMask[dst:dst+seqLen], mask[src:src+seqLen])
		copy(compactTypes[dst:dst+seqLen], types[src:src+seqLen])
	}
	session.telemetry.observeTokens(compactMask, batchSize, seqLen)
	return session.run(compactIDs, compactMask, compactTypes, batchSize, seqLen)
}
