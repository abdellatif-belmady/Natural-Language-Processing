---
hide:
  - navigation
---

## **Introduction**
***Abstractive Summarization*** est une tâche du Natural Language Processing (NLP) qui vise à générer un résumé concis d'un texte source. Contrairement au ***extractive summarization***, Abstractive Summarization ne se contente pas de copier les phrases importantes du texte source, mais peut également en créer de nouvelles qui sont pertinentes, ce qui peut être considéré comme une paraphrase. Abstractive Summarization donne lieu à un certain nombre d'applications dans différents domaines, des livres et de la littérature, à la science et à la R&D, à la recherche financière et à l'analyse de documents juridiques.

Jusqu'à présent, l'approche la plus récente et la plus efficace en matière de Abstractive Summarization consiste à utiliser des modèles de transformation spécifiquement adaptés à un ensemble de données de résumé. Dans cette étude, nous démontrons comment vous pouvez facilement résumer un texte à l'aide d'un modèle puissant en quelques étapes simples. Tout d'abord, nous utiliserons deux modèles qui sont déjà pré-entraînés, de sorte qu'aucun entrainnement supplémentaire n'est nécessaire, puis nous affinerons l'un de ces deux modèles sur notre base de données.

Sans plus attendre, commençons !

## **Importer les données**

```py
import pandas as pd
data = pd.read_json("/content/sample_data/AgrSmall.json")
data.head()
```

## **Utilisation de transformer `bart-large-cnn` & `t5-base`**

### **Installer la bibliothèque Transformers**
La bibliothèque que nous allons utiliser est Transformers par Huggingface.

Pour installer des transformateurs, il suffit d'exécuter cette cellule :

```py
pip install transformers
```
!!! note

    Transformers nécessite l'installation préalable de Pytorch. Si vous n'avez pas encore installé Pytorch, rendez-vous sur [le site officiel de Pytorch](https://pytorch.org/) et suivez les instructions pour l'installer.

### **Importer les bibliothèques**

Après avoir installé transformers avec succès, nous pouvons maintenant commencer à l'importer dans votre script Python. Nous pouvons également importer `os` afin de définir la variable d'environnement à utiliser par le GPU à l'étape suivante.

```py
from transformers import pipeline
import os
```

Maintenant, nous sommes prêts à sélectionner the summarization model à utiliser. Huggingface fournit deux summarization models puissants à utiliser : BART (bart-large-cnn) et t5 (t5-small, t5-base, t5-large, t5-3b, t5-11b). Pour en savoir plus sur ces modèles veuillez consulter leurs documents officiels ([document BART](https://arxiv.org/abs/1910.13461), [document t5](https://arxiv.org/abs/1910.10683)).


Pour utiliser le modèle BART, qui est formé sur le [CNN/Daily Mail News Dataset](https://www.tensorflow.org/datasets/catalog/cnn_dailymail), nous avons utilisés directement les paramètres par défaut via le module intégré Huggingface pipeline :

```py
summarizer = pipeline("summarization")
```

Pour utiliser le modèle t5 (par exemple t5-base), qui est entraîné sur [c4 Common Crawl web corpus](https://www.tensorflow.org/datasets/catalog/c4), nous avons procédé comme suit :

```py
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
```

Pour plus d'informations, veuillez vous référer à la [Huggingface documentation](https://huggingface.co/transformers/main_classes/pipelines.html#transformers.SummarizationPipeline).

### **Entrer le texte à résumer**

Maintenant que notre modèle est prêt, nous pouvons commencer à choisir le texte que nous voulons résumer. Nous proposons de choisir le premier abstract dans notre base de données :

Nous définissons notre variable :

```py
text = data["abstracts"][0]
print(text)
```

??? success "Output"
    Most people in rural areas in South Africa (SA) rely on untreated drinking groundwater sources and pit latrine sanitations. A minimum basic sanitation facility should enable safe and appropriate removal of human waste, and although pit latrines provide this, they are still contamination concerns. Pit latrine sludge in SA is mostly emptied and disposed off-site as waste or buried in-situ. Despite having knowledge of potential sludge benefits, most communities in SA are reluctant to use it. This research captured social perceptions regarding latrine sludge management in Monontsha village in the Free State Province of SA through key informant interviews and questionnaires. A key informant interview and questionnaire was done in Monontsha, SA. Eighty participants, representing 5% of all households, were selected. Water samples from four boreholes and four rivers were analyzed for faecal coliforms and E.coli bacteria. On average, five people in a household were sharing a pit latrine. Eighty-three percent disposed filled pit latrines while 17% resorted to closing the filled latrines. Outbreaks of diarrhoea (69%) and cholera (14%) were common. Sixty percent were willing to use treated faecal sludge in agriculture. The binary logistic regression model indicated that predictor variables significantly (p ˂ 0.05) described water quality, faecal sludge management, sludge application in agriculture and biochar adaption. Most drinking water sources in the study had detections ˂ 1 CFU/100 mL. It is therefore imperative to use both qualitative surveys and analytical data. Awareness can go a long way to motivate individuals to adopt to a new change. View Full-Text


### **Génération de résumé**

Enfin, nous pouvons commencer à résumer le texte entré. Ici, nous déclarons la longueur minimale et la longueur maximale que nous souhaitons pour la sortie du résumé, et nous désactivons également l'échantillonnage pour générer un résumé fixe. Nous pouvons le faire en exécutant la commande suivante :

```py
summary_text = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text)
```
Voilà ! Nous obtenons le résumé de notre texte :

??? success "Output"
    Most people in rural areas in South Africa rely on untreated drinking groundwater sources and pit latrine sanitations . Outbreaks of diarrhoea (69%) and cholera (14%) were common. Sixty percent were willing to use treated faecal sludge in agriculture .

## **Fine-tuning t5-base**





