# Analyzing-Movies-with-Object-Detection-AI
Huggingface offers thousands of trained AI models for object detection. Below is a neat example, on how to use such a model, in this case we use Facebook-detr-resnet-101 https://huggingface.co/facebook/detr-resnet-101 to detect objects in movies.
First of all, we extract a suitable amount of images from a movie, ffmpeg is very easy to use and reasonably quick. We store these frame00xy.jpg files in a folder. We will do inference with above pretrained model on these files.
The model is also able to detect where in the image a certain item/class is located, the code contains functions to mark these objects with boxes. For copyright reasons, we will not show any images of the movies we analyzed.

"Lost in Translation": Perhaps Sofia Coppola's best movie is set in mostly set in Tokyo, thus we can expect to find a lot of persons in the images. The main character Bob Harris is movie star working in Tokyo on ads for a (real) whisky brand, he likes to drink and does so regularly, so we can choose an appropriate class ('wine glass') to detect the whisky and cocktail glasses. We also find some dining tables, ties and bottles in the frames of this movie. In the end we can plot this info in a bar graph with the frames as bins (the x-axis). Clearly, even without knowing the movie (you should watch it, it's great..) and assuming everything works as advertised, we can assume, that there are some scenes with a drinking. Some scenes take place in the streets of Tokyo, thus also a lot of persons are detected (up to 25!) in some frames:

![Lost in Translation annotated analysis](https://github.com/user-attachments/assets/3516f01c-984f-4cf0-b728-5c13817e5ee3)

Clearly, this can be improved in terms of classes and number of frames. A problem is that in movies focus shifts, panning and intentional out-of-focus are style choices. This creates problems for any object detection software. If you know the movie, you can roughly follow the story line by looking at the graph. However, we could have chosen other classes, or used an entirely different model. This script is not complete and never will be. But perhaps you know what we are alluding to: Imagine a movie database like Kodi (there are others) filled with thousands of movies, you come home, want to watch a great movie, you forgot the title and just know som bits of the story line or other facts about the movie. A system like Kodi could be enhanced (maybe they are already working on something like this, IDK) with such AI object detection features.
Let's take a look at another movie where we can be sure to find some distinct classes, not just persons: 

"Four weddings and a Funeral": This mid-90s romcom features, as the title suggests, four weddings and a funeral. Thus, we can expect high scores of 'person', 'wine glass', 'tie' and 'bottle' in the footage:

![Four weddings and a Funeral annotated analysis](https://github.com/user-attachments/assets/d34a9248-ddb2-4548-8d75-5e6aba72be0b)

From the graph we can clearly discern the four weddings with the accumulated numbers of people, glasses and bottles. As Charles, more or less, cancels his own wedding, we can't find any glasses or bottles afterwards. Clearly, there's nothing to celebrate. 
