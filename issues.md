1. why is there two synthesize functions synthesize_neural(text, voice_id, speed, pitch) &  synthesize_from_tokens_neural(input_ids, style_vector, speed, model_id) 
2. why does this function even exists i thought python was managing the voices -  list_neural_voices()


1. how are this being used by python right now -> synthesize_neural & list_neural_voices

#### 1. make python not hardnoded list_neural_voices

#### 2. review this ORT_ENABLE_FP16 - Enable float16 optimization (set to "1" in onnx_engine.rs)   environement variable 

