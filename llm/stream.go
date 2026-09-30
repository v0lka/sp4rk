package llm

// StreamDelta is one incremental piece of a streamed completion: the assistant
// text that arrived since the previous delta and/or a reasoning
// (chain-of-thought) delta. An empty field carries no new data for that stream.
//
// Streaming is opt-in through ChatRequest.DeltaSink (see the field doc). A
// provider invokes that callback with successive deltas as they arrive off the
// wire and then returns the fully-assembled *ChatResponse exactly as the
// non-streaming path would. Streaming therefore changes only the DELIVERY
// TIMING of the text; the final response — tool calls, usage, stop reason — is
// identical to the synchronous result.
type StreamDelta struct {
	// Text is assistant content text that arrived since the previous delta.
	Text string
	// Reasoning is reasoning / chain-of-thought text that arrived since the
	// previous delta (empty for models that do not surface reasoning).
	Reasoning string
}
