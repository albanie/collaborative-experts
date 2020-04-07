def sent_feat (text, feat_type):
    
    if feat_type =='w2v':
        import gensim
        import numpy as np
        model = gensim.models.KeyedVectors.load_word2vec_format('/scratch/shared/slow/yangl/w2v/GoogleNews-vectors-negative300.bin', binary=True)
        final_feats=[]
        for word in (text.split(' ')):
            if (word !='a') and (word in model.vocab):
                final_feats.append(model.get_vector(word))

        final_feats = np.asarray(final_feats)
    
    elif feat_type == 'openai':
        import json
        import torch
        from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
        import logging
        
        logging.basicConfig(level=logging.INFO)

        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

        # Tokenized input
        #text = "Who was Jim Henson ? Jim Henson was a puppeteer"
        model = OpenAIGPTModel.from_pretrained('openai-gpt')
        model.eval()
        model.to('cuda')
        
        tokenized_text = tokenizer.tokenize(text)

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])


        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')


        # Predict hidden states features for each layer
        with torch.no_grad():
            hidden_states = model(tokens_tensor)
            final_feats = hidden_states[0].cpu().numpy()


    else:
        print ('Unrecognised FEAT_TYPE.')
        
    return final_feats

if __name__ == '__main__':
    
    query_sent = 'a cartoon animals runs through an ice cave in a video game'
    print ("Query: {}".format(query_sent))

    print ("FEAT_TYPE can be selected from ['w2v', 'openai']")

    w2v_feats = sent_feat(query_sent,'w2v')
    print ("word2vec shape is: {}".format(w2v_feats.shape))

    openai_feats = sent_feat(query_sent,'openai')
    print ("openai shape is: {}".format(openai_feats.shape))