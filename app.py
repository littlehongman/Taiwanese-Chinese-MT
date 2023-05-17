import streamlit as st
import time
from projects.translate.model import load_model_and_vocab, translate_sentence_beam_search

st.set_page_config(page_title="Machine Translation", page_icon="🎈", layout="wide")

# Load model
@st.cache_resource()
def load_model():
    return load_model_and_vocab()

with st.spinner('Loading....'):
    SRC_vocab, TRG_vocab, model = load_model()


# CSS Style
hide_component = '''

    <style>
        button[title="View fullscreen"]{visibility: hidden;}
        footer {visibility: hidden;}
        .css-fk4es0 {visibility: hidden;}
        
        .css-1ab3oty {border: none !important;}
        .element-container{opacity:1 !important}
        [href="#nlp-demo"] { display: none;}
    </style>
'''
st.markdown(hide_component, unsafe_allow_html=True)
            

    
st.header("**Taiwanese-Chinese Machine Translation**")
#st.markdown("<p style='font-size:120%; font-weight:bold; text-align:left'>Taiwanese-Chinese Machine Translation</h5>", unsafe_allow_html=True)
st.markdown(
    "This app generates Machine Translation from <strong>Taiwanese (Hokkien)</strong> to <strong>Traditional Chinese</strong> characters using the [Transformer](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb) architecture. We also implemented Beam Search and Length penalty to improve performance."
,unsafe_allow_html=True)


# Input Are
text = st.text_area("Enter Text", "keng2-hong kong2 hiam5-hoan7 choan-bun5 ti7 hak8-hau7 hu3-kin7 ， chhoe7 hak8-seng-e5 sok-sia3 ha7-chhiu2 thau-theh8 。")
beam_width = st.slider('Beam Width', 3, 100, 5)
alpha = st.slider('Alpha', 0.1, 1.0, 0.7)

# Submit
submitted = st.button("Translate")
if submitted:
    with st.spinner('Wait for it...'):
        seqs = translate_sentence_beam_search(text, SRC_vocab, TRG_vocab, model, beam_width, alpha)
    #st.success(result)
    st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">', unsafe_allow_html=True)
    
    #print(seqs)
    #print(scores)

    st.markdown(
        f'''
        <div class="card" >
            <div class="card-body">
                <p style='font-size:120%; font-weight:bold' class="card-title">Result</p>
                <p class="card-text">Top 1：{"".join(seqs[0])}</p>
                <p class="card-text">Top 2：{"".join(seqs[1])}</p>
                <p class="card-text">Top 3：{"".join(seqs[2])}</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("<br/><br/>", unsafe_allow_html=True)
with st.expander("References"):
    st.write("[Beam Search](https://youtu.be/RLWuzLLSIgw)")
    st.write("[Refining Beam Search](https://youtu.be/gb__z7LlN_4)")