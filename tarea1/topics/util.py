import csv
import pandas as pd
import numpy as np

import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import plotly.express as px
import json

import spacy
import string 

nlp = spacy.load("en_core_web_sm")

from gensim import corpora
from gensim import models
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from gensim.matutils import corpus2csc

pd.set_option('max_colwidth', 200)

from collections import defaultdict

def read_network(file_path):
    f = open(file_path, encoding='utf-8-sig') # delete descripcion:
    network = json.load(f)["network"]
    network_items = pd.DataFrame(network["items"])
    network_items["DOI"] = network_items["url"].apply(lambda x: x.replace("https://doi.org/", "").lower() if str(x) != "nan" else np.nan)
    network_items["Citations"] = network_items["weights"].apply(lambda x: x["Citations"])
    network_items["Links"] = network_items["weights"].apply(lambda x: x["Links"])
    network_items["id"] = network_items["id"]-1
    return network_items

def read_bibliography(file_path):
    df_scopus = pd.read_csv(file_path)
    df_scopus = df_scopus.reset_index().rename(columns={"index": "id"})
    return df_scopus

def get_docs(df_net, df_biblio):
    relevant_columns = {
        "LSA": "Title  Abstract  Author Keywords  Index Keywords".split("  "),
        "others": "Year  Authors".split("  ")
    }
    df_docs = df_biblio[["id"]+relevant_columns["LSA"]+relevant_columns["others"]]
    df_docs = df_docs.merge(df_net, how="inner", on="id")
    df_docs["Abstract"] = df_docs["Abstract"].replace({"[No abstract available]": np.nan})
    df_docs = df_docs.fillna(" ")
    df_docs["doc"] = df_docs.apply(lambda x: ". ".join(x[relevant_columns["LSA"]].values), axis=1)
    return df_docs

def plot_frec_clusters(df_docs):
    num_clusters = df_docs["cluster"].unique().shape[0]
    table_hist = pd.DataFrame()
    table_hist["#Docs"] = df_docs["cluster"].value_counts().values
    table_hist["Cluster"] = [str(i) for i in range(1, num_clusters+1)]
    color_discrete_sequence = ['#d63f4b', 
                               "#58bd5b", 
                               "#5599c3", 
                               "#bcbe4f", 
                               "#996fc0", 
                               "#5cccd9"][:num_clusters]
    fig = px.bar(table_hist, 
                 x="Cluster", 
                 y="#Docs", 
                 color="Cluster", 
                 title="#Docs per cluster (LinLog/mod.)", 
                 color_discrete_sequence=color_discrete_sequence)
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    fig.show()

def plot_docs_per_year(df_biblio):
    fig = px.histogram(df_biblio, x="Year", text_auto=True)
    fig.update_layout(
        title="Histogram of #Docs per year"
    )
    fig.show(renderer="notebook")
    
def plot_docs_per_source(df_biblio):
    frec_sources = pd.DataFrame(df_biblio["Source title"].value_counts()).reset_index().iloc[:25,:].rename(columns={"index": "Source title", "Source title": "#Docs"})
    fig = px.bar(frec_sources, x="Source title", y="#Docs", text_auto=True)
    fig.update_layout(
        title="Histogram of #Docs per source"
    )
    fig.show(renderer="notebook")

replace_acronyms = {
    "BHCP": ["backward", "heat", "conduction", "problem"],
    "LGSM": ["lie", "group", "shoot", "method"],
    "GPS": ["group", "preserve", "scheme"],
    "TCC": ["thermal", "contact", "conductance"],
    "IHCP": ["inverse", "heat", "conduction", "problem"],
    "CGM": ["conjugate", "gradient", "method"],
    "PSO": ["particle", "swarm", "optimization"],
    "IHTP": ["inverse", "heat", "transfer", "problem"], 
    "BEM": ["boundary", "element", "method"],
}
replace_tokens = {
}
ignore_tokens = []

def preprocess(text, n_gram_range = (1, 2)):
    tokens, matching = [], []    
    
    for k, v in replace_tokens.items():
        if k in text: text.replace(k, v)
            
    i, j = 0, 0
    for token in nlp(text):
        val = token.text
        if val not in string.punctuation+"'":
            if not token.is_stop:
                if "x" in token.shape_.lower():
                    p_token = token.lemma_.lower()
                    if p_token not in ignore_tokens:
                        if p_token in replace_acronyms.keys():
                            new_tokens = replace_acronyms[p_token]
                            tokens += new_tokens
                            matching += [(i, j+k) for k in range(len(new_tokens))]
                            j += len(new_tokens)
                        else:
                            tokens.append(p_token)
                            matching.append((i, j))
                            j += 1
        i += 1
    if 2 in n_gram_range:     
        num_tokens = len(tokens)  
        bi_tokens = []
        for k in range(len(tokens)-1):
            bi_tokens.append(f"{tokens[k]}_{tokens[k+1]}")
            matching.append(((matching[k][0], matching[k+1][0]), num_tokens+k))
        return {"doc_clean": tokens+bi_tokens, "matching": matching}
    else:
        return {"doc_clean": tokens, "matching": matching}

def add_preprocess(df_docs):
    if "doc_clean" not in df_docs:
        df_docs['preprocess'] = df_docs['doc'].apply(lambda x: preprocess(x))
        df_docs['doc_clean'] = df_docs['preprocess'].apply(lambda x: x["doc_clean"])
        df_docs['matching'] = df_docs['preprocess'].apply(lambda x: x["matching"])
    pass

def LSA(df_docs, cluster_id, num_topics):
    df_cluster = df_docs[df_docs["cluster"] == cluster_id]
    corpus = df_cluster['doc_clean']
    dictionary = corpora.Dictionary(corpus)

    bow = [dictionary.doc2bow(text) for text in corpus]

    tfidf = models.TfidfModel(bow)
    corpus_tfidf = tfidf[bow]

    lsi = LsiModel(
        corpus_tfidf, 
        num_topics=num_topics, 
        id2word=dictionary, 
        random_seed=2022
    )
    return {
        "lsi": lsi, 
        "corpus_tfidf": corpus_tfidf,
        "num_topics": num_topics, 
        "df_cluster": df_cluster, 
        "cluster_id": cluster_id
    }

def plot_top_eigenvalues(lsa, cluster_id=1, k=25):
    corpus_csc = corpus2csc(lsa["corpus_tfidf"])
    vals, _ = eigs(corpus_csc @ corpus_csc.T, k=k)
    vals = np.real(vals)

    top_eigs_vals = pd.DataFrame(vals).reset_index()
    top_eigs_vals["index"] += 1
    top_eigs_vals["index"] = top_eigs_vals["index"].astype(str)
    top_eigs_vals = top_eigs_vals.rename(columns={0: "Value", "index": "N-th eigenvalue"})

    fig = px.bar(
        top_eigs_vals, 
        x="N-th eigenvalue", 
        y="Value", 
        title=f"Top {k} eigenvalues (Cluster {cluster_id})"
    )
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    fig.show(renderer="notebook")
    
def get_topics(lsa, num_std_u=1, num_std_v=1): #u: terms, v: document
    abs_u = abs(lsa["lsi"].projection.u)
    tresh_u = np.mean(abs_u, axis=0) + num_std_u * np.std(abs_u, axis=0)

    index_u_per_factor = np.where(abs_u>tresh_u)
    term_per_factor = csr_matrix(
        (
            abs_u[index_u_per_factor[0], index_u_per_factor[1]], 
            (index_u_per_factor[0], index_u_per_factor[1])
        ), 
        shape=abs_u.shape
    )

    num_terms = int(lsa["lsi"].num_terms)
    num_docs = int(lsa["lsi"].docs_processed)
    corpus_csc = corpus2csc(lsa["corpus_tfidf"], num_terms=num_terms, num_docs=num_docs)

    doc_per_factor = corpus_csc.T @ term_per_factor
    doc_per_factor = doc_per_factor / np.sum(doc_per_factor, axis=1)
    doc_per_factor = np.asarray(doc_per_factor)

    tresh_v = np.mean(doc_per_factor, axis=0) + num_std_v * np.std(doc_per_factor, axis=0)
    index_v_per_factor = np.where(doc_per_factor>tresh_v)

    index_per_factor = {
        "u": index_u_per_factor, 
        "v": index_v_per_factor
    }
    return {
        "term_per_factor": term_per_factor, 
        "doc_per_factor": doc_per_factor, 
        "index_per_factor": index_per_factor
    }

def get_table_hist_per_topic(index_per_factor):
    index_v_per_factor, index_u_per_factor = index_per_factor["v"], index_per_factor["u"]
    
    table_hist_per_topic = pd.DataFrame(pd.Series(index_v_per_factor[1].astype(str)).value_counts())
    table_hist_per_topic = table_hist_per_topic.reset_index().rename(columns={"index": "Topic", 0: "#Docs"})
    
    aux = pd.DataFrame(pd.Series(index_u_per_factor[1].astype(str)).value_counts())
    aux = aux.reset_index().rename(columns={"index": "Topic", 0: "#Terms"})
    table_hist_per_topic = table_hist_per_topic.merge(aux, on="Topic", how="left")
    
    table_hist_per_topic = table_hist_per_topic.sort_values("Topic")
    return table_hist_per_topic

def plot_topics(table_hist_per_topic, cluster_id, by):
    fig = px.bar(
        table_hist_per_topic, 
        color="Topic", 
        x="Topic", 
        y=f"#{by}", 
        title=f"#{by} per topic (Cluster {cluster_id})"
    )
    fig.update_layout(
        font=dict(
            size=18
        )
    )
    fig.show()
    
def get_strengths(lsa, term_per_factor, num_terms=10):
    strengths = [] 
    for factor in range(term_per_factor.shape[1]):
        id_terms_f = np.argsort(term_per_factor[:, factor].toarray().flatten())[::-1][:num_terms]
        strenght_f = term_per_factor[id_terms_f, factor].toarray().flatten()
        terms_f = [lsa["lsi"].id2word[ix] for ix in id_terms_f]
        for t, s in zip(terms_f, strenght_f):
            o = {
                'Topic': str(factor), 
                'Term': t, 
                'Strength': s
            }
            strengths.append(o)
    return strengths

def plot_strength_term_per_topic(strengths, cluster_id):
    table_strength = pd.DataFrame(strengths)
    table_strength = table_strength.sort_values(["Topic", "Strength"], ascending=[True, False])

    fig = px.bar(
        table_strength, 
        color="Topic", 
        y="Term", 
        x="Strength", 
        title=f"Strength of topics terms (Cluster {cluster_id})"
    )
    fig.update_layout(
        font=dict(
            size=10
        ),
        height=800,
        width=600
    )
    fig.show()
    
def plot_strength_doc_per_topic(weighted_cluster, cluster_id):
    
    topic_colnames = [c for c in weighted_cluster.columns if "topic_" in c]
    num_topics = len(topic_colnames)
    
    weighted_cluster = weighted_cluster.sort_values("Year")
    
    df = weighted_cluster[["label"]+topic_colnames]
    table_strength = pd.melt(
        df, 
        id_vars=['label'], 
        value_vars=topic_colnames, 
        var_name="Topic", 
        value_name="Strength"
    ).rename(columns={"label": "Doc"})
    
    table_strength["Topic"] = table_strength["Topic"].apply(lambda x: x.replace("topic_", ""))
    color_map = {
        str(t): px.colors.qualitative.Plotly[t]
        for t in range(num_topics)
    }
    magic_number = 16.85
    fig = px.bar(
            table_strength, 
            color="Topic", 
            x="Doc", 
            y="Strength", 
            title=f"Strength of topics docs (Cluster {cluster_id})",
            color_discrete_map=color_map
        )
    fig.update_layout(
            font=dict(
                size=10
            ),
            height=600,
            width=int(magic_number*df.shape[0])
        )
    fig.show()
    
def get_terms_table(table_hist_per_topic, strengths):
    df_terms = pd.DataFrame(strengths).drop(columns="Strength").groupby("Topic").aggregate(
        lambda x: ", ".join(x).replace("_", " ")
    )
    df_terms = df_terms.reset_index().rename(columns={"Term": "Top 10 terms"})
    df_terms = df_terms.merge(table_hist_per_topic, on="Topic", how="left")
    df_terms = df_terms["Topic  Top 10 terms  #Terms  #Docs".split("  ")]
    return df_terms.set_index("Topic")

def get_group_table(network_items):
    table_summary = pd.DataFrame(
        {
            "#Docs": network_items["cluster"].fillna(0).value_counts().values,
            "Group": ["Others", "Dense component"]
        }
    ).set_index("Group")
    return table_summary

def get_links_stats_table(network_items):
    stats = [
        ("sum", sum), 
        ("mean", np.mean), 
        ("std", np.std), 
        ("min", min), 
        ("25%", lambda x: np.quantile(x, q=0.25)), 
        ("50%", lambda x: np.quantile(x, q=0.50)), 
        ("75%", lambda x: np.quantile(x, q=0.75)),
        ("max", max)
    ]
    table_stats = network_items["cluster Links".split()].fillna({"cluster": 0}).groupby("cluster").aggregate(stats)
    table_stats.index = ["Others", "Dense component"]
    stats_overall = network_items["cluster Links".split()].fillna({"cluster": 1}).groupby("cluster").aggregate(stats)
    stats_overall.index = ["All"]
    table_stats = pd.concat([stats_overall, table_stats], axis=0)
    table_stats.index.name = "Docs"
    return table_stats

def get_top_table_clusters(df_docs, network_items, num_clusters, by, min_val):
    id_top = network_items[network_items[by] >= min_val]["id"]
    top = df_docs[df_docs["id"].isin(id_top)]
    table_top = top[top["cluster"] == 1].sort_values("Year")
    for t in range(2, num_clusters+1):
        table_top = pd.concat([table_top, top[top["cluster"] == t].sort_values("Year")])
    table_top = table_top[f"cluster Year Authors Title {by}".split()].rename(columns={"cluster": "Cluster"})
    table_top[by] = table_top[by].astype(int)
    return table_top

def get_weighted_cluster(lsa, topics):
    index_per_factor = topics["index_per_factor"]
    doc_per_factor = topics["doc_per_factor"]
    doc_topics = []
    last_ix = ""
    for ix, t in zip(*index_per_factor["v"]):
        if last_ix != ix:
            o = {"id": lsa["df_cluster"].id.iloc[ix], f"topic_{t}": doc_per_factor[ix, t]}
            doc_topics.append(o)
        else:
            o = doc_topics.pop()
            o[f"topic_{t}"] = doc_per_factor[ix, t]
            doc_topics.append(o)
        last_ix = ix
    return lsa["df_cluster"].merge(pd.DataFrame(doc_topics), on="id", how="outer").fillna(0)

def get_top_table_topics(weighted_cluster, num_topics, by, min_val):
    id_top = weighted_cluster[weighted_cluster[by] >= min_val]["id"]
    top = weighted_cluster[weighted_cluster["id"].isin(id_top.values)]
    table_top = top[top["topic_0"] > 0].sort_values("Year")
    table_top = table_top.rename(columns={"topic_0": "Strenght"})
    table_top["Topic"] = "0"
    for t in range(1, num_topics):
        sub_table_top = top[top[f"topic_{t}"] > 0].sort_values("Year")
        sub_table_top = sub_table_top.rename(columns={f"topic_{t}": "Strenght"})
        sub_table_top["Topic"] = str(t)
        table_top = pd.concat([table_top, sub_table_top])
        table_top = table_top[f"Topic Strenght Year Authors Title {by}".split()]
    table_top[by] = table_top[by].astype(int)
    return table_top

from IPython.core.display import HTML, display

def display_topics(lsa, topics, label):
    label_docs = list(lsa["df_cluster"]["label"])
    ix_doc = label_docs.index(label)

    rgb_topics = [
        '0,0,255', 
        "255,0,0", 
        "0,255,0", 
        "204,0,204", 
        "255,204,0", 
        "0,204,255"
    ]

    corpus_csc = corpus2csc(lsa["corpus_tfidf"])
    term_per_factor = topics["term_per_factor"]

    tfidf_doc = corpus_csc[:, ix_doc]
    doc = lsa["df_cluster"].iloc[ix_doc]["doc"]
    doc_clean = lsa["df_cluster"].iloc[ix_doc]["doc_clean"]
    matching = lsa["df_cluster"].iloc[ix_doc]["matching"]

    token2id = lsa["lsi"].id2word.token2id

    token_ids = [
        token2id[token]
        for token in doc_clean
    ]
    doc_words = nlp(doc)

    title = lsa["df_cluster"].iloc[ix_doc]["Title"]
    abstract = lsa["df_cluster"].iloc[ix_doc]["Abstract"]
    authkwds = lsa["df_cluster"].iloc[ix_doc]["Author Keywords"]
    indexkwds = lsa["df_cluster"].iloc[ix_doc]["Index Keywords"]

    title_words = nlp(title)
    abstract_words = nlp(abstract)
    authkwds_words = nlp(authkwds)
    indexkwds_words = nlp(indexkwds)

    num_topics = term_per_factor.shape[1]
    tokens_ids_importance = {}
    for ix_pos_topic in range(num_topics):
        term_pos_per_factor = term_per_factor[:, ix_pos_topic].A
        tfidf_pos_factor = (tfidf_doc.A * term_pos_per_factor).ravel()
        tokens_ids_pos = np.where(tfidf_pos_factor>0)[0]
        for k in tokens_ids_pos:
            if k not in tokens_ids_importance.keys():
                tokens_ids_importance[k] = np.zeros(num_topics)
            tokens_ids_importance[k][ix_pos_topic] = tfidf_pos_factor[k]

    tokens_importance = [
        tokens_ids_importance[token_id] 
        if token_id in tokens_ids_importance.keys() 
        else np.zeros(num_topics) for token_id in token_ids
    ]

    grams_importance = {}
    for pair in matching:
        i, j = pair[0], pair[1]
        if type(i) != tuple:
            if i not in grams_importance.keys():
                grams_importance[i] = np.zeros(num_topics)
            grams_importance[i] += tokens_importance[j]
        else:
            for ik in i:
                if ik not in grams_importance.keys():
                    grams_importance[ik] = np.zeros(num_topics)
                grams_importance[ik] += tokens_importance[j]

    word_importance = []
    for k, w in enumerate(doc_words):
        if k in grams_importance.keys():
            most_important_topic = np.argmax(grams_importance[k])
            word_importance.append(
                (
                    w, 
                    most_important_topic, 
                    grams_importance[k][most_important_topic]
                )
            )
        else:
            word_importance.append((w, -1, 0))

    tresh = 0.001
    rgb = lambda x: rgb_topics[x]
    alpha = lambda x: abs(x) * 10 if abs(x)>tresh else 0

#     doc_word_marks = [
#             f'<mark style="background-color:rgba({rgb(colour)},{alpha(attr)})">{word}</mark>'
#             for word, colour, attr in word_importance
#         ]

#     return display(HTML('<p>' + ' '.join(doc_word_marks) + '</p>'))

    title_word_marks = [
            f'<mark style="background-color:rgba({rgb(colour)},{alpha(attr)})">{word}</mark>'
            for word, colour, attr in word_importance[:len(title_words)]
        ]

    abstract_word_marks = [
            f'<mark style="background-color:rgba({rgb(colour)},{alpha(attr)})">{word}</mark>'
            for word, colour, attr in word_importance[len(title_words)+1:len(title_words)+len(abstract_words)+1]
        ]

    authkwds_word_marks = [
            f'<mark style="background-color:rgba({rgb(colour)},{alpha(attr)})">{word}</mark>'
            for word, colour, attr in word_importance[len(title_words)+len(abstract_words)+1+int(len(abstract_words)==1):len(title_words)+len(abstract_words)+len(authkwds_words)+1]
        ] 

    indexkwds_word_marks = [
            f'<mark style="background-color:rgba({rgb(colour)},{alpha(attr)})">{word}</mark>'
            for word, colour, attr in word_importance[len(title_words)+len(abstract_words)+len(authkwds_words)+int(len(abstract_words)==1)+2:]
        ]

    return display(
        HTML(
            "<b> Title: </b>"+'<p>' + ' '.join(title_word_marks) + '</p>'+ 
            "<b> Abstract: </b>"+'<p>'+' '.join(abstract_word_marks) + '</p>'+
            "<b> Author keywords: </b>"+'<p>'+' '.join(authkwds_word_marks) + '</p>'+
            "<b> Index keywords: </b>"+'<p>'+' '.join(indexkwds_word_marks) + '</p>'
        )
    )