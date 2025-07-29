from transformers import PreTrainedTokenizerBase

def get_multimodal_keys_from_processor(processor) -> list[str]:
    '''
    Get keys of the multimodal data that can be used as model inputs.
    This will be used in the data_processor function to determine which keys to use as model inputs.
    '''
    if isinstance(processor, PreTrainedTokenizerBase):
        return []

    all_keys = set()
    if hasattr(processor, "image_processor"):
        all_keys.update(processor.image_processor.model_input_names)
    if hasattr(processor, "video_processor"):
        all_keys.update(processor.video_processor.model_input_names)
    if hasattr(processor, "feature_extractor"):
        all_keys.update(processor.feature_extractor.model_input_names)
    # remove keys that are already in the tokenizer
    all_keys.difference_update(set(processor.tokenizer.model_input_names))
    return list(all_keys)