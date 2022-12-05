import streamlit as st
from Experiments import BaseExp, get_random_sample_latent_diffs
from Utils import get_label_idx, load_models

if 'started' not in st.session_state:
    device='cuda'
    st.title("Latent Study Analysis")
    exp = BaseExp(device, batch_size=32)
    model_dct = load_models(device)
    test_idx_dct = get_label_idx(exp.test_loader.dataset)
    st.session_state['device'] = device
    st.session_state['exp'] = exp
    st.session_state['model_dct'] = model_dct
    st.session_state['test_idx_dct'] = test_idx_dct

if 'retrieved_latent_diffs' not in st.session_state:
    pass

def get_latent_diffs_dct():
    latent_diffs_dct = {}
    model_dct = st.session_state['model_dct']
    test_idx_dct = st.session_state['test_idx_dct']
    device = st.session_state['device']
    for i in range(10):
        latent_diffs_dct[i] = get_random_sample_latent_diffs(i, model_dct, test_idx_dct,
                                                             device, )
