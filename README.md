Flying Farmer is an application to help users or farmers to identify diseases such as: Brown Spot, Hispa and Leaf Blast. 

In this application, we use the pretrained model of resnet 101 by pytorch: https://pytorch.org/vision/stable/models.html#torchvision.models.resnet101  
or it's called as Deep Residual Learning for Image Recognition: https://arxiv.org/pdf/1512.03385.pdf . 

To train our CNN-model, we need to provide enormous dataset of rice diceases photos from Kaggle: https://www.kaggle.com/minhhuy2810/rice-diseases-image-dataset
In this data provider, there are 4 different conditions of rice which are brown spot, hispa, leaf blast and healthy. By this dataset we are able to train our image-classifier model 
to help our program to identify the photo which contains rice leaf. 

How to use our program:
1. First, you can clone our program from GitHub.
2. Then, you must install all important python moduls such as pandas, pytorch, torchvision, PIL Image and dash by plotly.
3. With Dash framework, the program will initialize the code, then provides a temporar server which will load the code and the local IP-address to open the local server in a browser.
4. Copy the IP-address, which shows up in compiler after running the code. Then open your browser and paste the copied IP-address. After that, the UI will show up.
5. Now you can start to use our program by uploading the rice photo.

If you find the problem to using our program, you can contact us by our LinkedIn.
Azmi Hasan: https://www.linkedin.com/in/azmihasan/
Rama W. B.: https://www.linkedin.com/in/rama-widyadhana-bhagaskoro-7253581b2/
M. Farhan F.: https://www.linkedin.com/in/muhammad-farhan-fadhilah/
