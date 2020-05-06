import json as js
import numpy as np
import nltk
import unicodedata
from nltk.stem.rslp import RSLPStemmer
import tflearn as tfl
import tensorflow as tf
import random
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import requests

#pega dados da api
def getAll():
  r = requests.get('https://covid19-brazil-api.now.sh/api/report/v1/brazil')
  re = r.json()
  return re['data']

#cria o audio e salva na pasta audios e executa
def cria_audio(text,lang):
  	print('falando')
  	tts = gTTS(text,lang=lang)
  	tts.save('audios/text.mp3')
  	playsound('audios/text.mp3')

def ouvir_microfone():
	microfone = sr.Recognizer()
	with sr.Microphone() as source:
		microfone.adjust_for_ambient_noise(source)
		print("ouvindo...")
		audio = microfone.listen(source)

	try:
		print('reconhecendo')
		phrase = microfone.recognize_google(audio,language='pt-BR')
		print('ok')

	except sr.UnknownValueError:
		return "Não entendi"

	return phrase

#abre o json
with open("intents.json") as file:
  data = js.load(file)

#tokeniza as palavras
palavras = []
intencoes = []
sentencas = []
saidas = []

# pega intenção por intenção
for intent in data["intents"]:
  
  tag = intent['tag'] 

  if tag not in intencoes:
     intencoes.append(tag)

  for pattern in intent["patterns"]:
    wrds = nltk.word_tokenize(pattern, language='portuguese')
    palavras.extend(wrds)
    sentencas.append(wrds)
    saidas.append(tag)


#remove palavras e pontos não importantes
p = []
for i in palavras:
  i = ''.join(ch for ch in unicodedata.normalize('NFKD', i) 
    if not unicodedata.combining(ch))
  p.append(i)

palavras = p

#stemming
stemer = RSLPStemmer()

stemmed_words = [stemer.stem(w.lower()) for w in palavras]
stemmed_words = sorted(list(set(stemmed_words)))

#bag of words
training = []
output = []
# criando um array preenchido com 0
outputEmpty = [0 for _ in range(len(intencoes))]

for x, frase in enumerate(sentencas):
  bag = []
  wds = [stemer.stem(k.lower()) for k in frase]
  for w in stemmed_words:
    if w in wds:
      bag.append(1)
    else:
      bag.append(0)

  outputRow = outputEmpty[:]
  outputRow[intencoes.index(saidas[x])] = 1

  training.append(bag)  
  output.append(outputRow)

#rede neural
training = np.array(training)
output = np.array(output)

tf.reset_default_graph()

# camada de entrada
net = tfl.input_data(shape=[None, len(training[0])])
# oito neuronios por camada oculta
net = tfl.fully_connected(net, 16)
# camada de saida
net = tfl.fully_connected(net, len(output[0]), activation="softmax")
# 
net = tfl.regression(net)

# criando o modelo
model = tfl.DNN(net)

#treinando modelo
model.fit(training, output, n_epoch=200, batch_size=16, show_metric=False)
model.save("./model/model.chatbot30G")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


def chat():
    print("Esse é o bot de teste! Converse com ele")
    Online = True
    while Online:
        inp = ouvir_microfone()
        bag_usuario = bag_of_words(inp, stemmed_words)
        results = model.predict([bag_usuario])

        valor = validate(results)
        if (valor != True):
          cria_audio("não entendi sua pergunta, tente novamente",'pt-br')
        else:
          results_index = np.argmax(results)
          tag = intencoes[results_index]

          for tg in data["intents"]:
              if tg['tag'] == tag:
                  if tag == "mortos":
                    responses = tg['responses']
                    number = api_data['deaths']
                    cria_audio(random.choice(responses),'pt-br')
                    cria_audio(str(number),'pt-br')
                  elif tag == "infectados":
                    responses = tg['responses']
                    number = api_data['confirmed']
                    cria_audio(random.choice(responses),'pt-br')
                    cria_audio(str(number),'pt-br')
                  elif tag == "recuperados":
                    responses = tg['responses']
                    number = api_data['recovered']
                    cria_audio(random.choice(responses),'pt-br')
                    cria_audio(str(number),'pt-br')
                  else:
                    responses = tg['responses']
                    cria_audio(random.choice(responses),'pt-br')
          if tag == "ate-mais":
            Online = False

def validate(prob):
   maiorResul = prob.max()
   print(maiorResul)
   if maiorResul > 0.23:
     return True
   else:
     return False


api_data = getAll()
chat()







