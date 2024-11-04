# Analyzing-Movies-with-Object-Detection-AI
Huggingface offers thousands of trained AI models for object detection. Below is a neat example, on how to use such a model, in this case we use Facebook-detr-resnet-101 https://huggingface.co/facebook/detr-resnet-101 to detect objects in movies.
First of all, we extract a suitable amount of images from a movie, ffmpeg is very easy to use and reasonably quick. We store these frame00xy.jpg files in a folder. We will do inference with above pretrained model on these files.
The model is also able to detect where in the image a certain item/class is located, the code contains functions to mark these objects with boxes. For copyright reasons, we will not show any images of the movies we analyzed.
"Lost in Translation": Perhaps Sofia Coppola's best movie is set in mostly set in Tokyo, thus we can expect to find a lot of persons in the images. The main character Bob Harris is movie star working in Tokyo on ads for a (real) whisky brand, he likes to drink and does so regularly, so we can choose an appropriate class ('wine glass') to detect the whisky and cocktail glasses. We also find some dining tables, ties and bottles in the frames of this movie. In the end we can plot this info in a bar graph with the frames as bins (the x-axis). Clearly, even without knowing the movie (you should watch it, it's great..) and assuming everything works as advertised, we can assume, that there are some scenes with a drinking. Some scenes take place in the streets of Tokyo, thus also a lot of persons are detected (up to 25!) in some frames:

![Lost in Translation annotated analysis](https://github.com/user-attachments/assets/3516f01c-984f-4cf0-b728-5c13817e5ee3)


