# Building a LSTM Encoder-Deconder using PyTorch to make Sequence-to-Sequence Predictions

## Requirements 
- Python 3 
- PyTorch

## 1 Overview 
There are many instances where we would like to predict how a time series will behave in the future. For example, we may be interested in predicting how many views a web page will receive. Given a time series of past viewership, what will the viewership be in a future time interval? Other time series we might like to predict the future values of include weather conditions (temperature, humidity, etc.), power usage, and traffic volume. In these examples, the past values of the time series can influence future values. It is therefore important to choose a modelling approach that can take these time dependencies into account. The Long Short-Term Memory (LSTM) neural network is a natural choice because it can extract important information about the time series over long time intervals. 

<p align="center">
  <img src="figures/hawking.jpg" width="900">
    <br>
 <em> <font size = "4"> Forcasting web page traffic is a time series prediction problem. Using past viewership, can we  <br> predict how many times Stephen Hawking's Wikipedia page will be viewed in the future? </font> </em>  
</p>

In this project, we will build special type of LSTM neural network: the LSTM encoder-decoder. This particular architecture enables us to make sequence-to-sequence predictions. In the web traffic example, a sequence-to-sequence prediction would be providing the network with 20 days of past viewership and predicting the next 5 days of viewership. The LSTM encoder-decoder consists of two LSTMs. The first LSTM, or the encoder, processes an input sequence and generates an encoded state. The encoded state summarizes the informaiton in the input sequence. The second LSTM, or the decoder, uses the encoded state to produce an output sequence. The LSTM encoder-decoder architecture is shown below. 

<p align="center">
  <img src="figures/encoder_decoder.png" width="700">
    <br>
 <em> <font size = "4"> The LSTM encoder generates an encoded state that summarizes the information in the input sequence. The LSTM decoder takes the encoded state and uses it to produce an output sequence.  </font> </em>  
</p>

Here, we will focus on a simple example where we use the LSTM encoder-decoder on synthetic data. We will fit a noisy sinusoidal curve. In Sec. 2, we prepare the time series datatset to feed into our LSTM encoder-decoder. In Secon 3. we build....
