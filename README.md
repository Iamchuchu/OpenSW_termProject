# Emotion Detection with OpenCV

The code is a program that reads images from a webcam or video file, detects and visually displays emotions around your face by detecting faces and objects. The code is written in Python's language, using different libraries and modules.

Dana Kim, YeSeo Kim, and DongHoon No are in charge of code production, and SeongYun Chu manages GitHub

---

## Description
The resulting (real_final_code.py) code is a program that contains two main functions.  
### 1. The first function is the recognition function of people and objects.  
If a person or object is recognized in the picture, a square box will be drawn around the recognized object.  
In order to accurately distinguish each object, the colors used are all set differently at random.  
  
### 2. The second function is to recognize people's emotions.  
This second part code only works when a person is recognized, and it detects people's emotions by classifying them into a total of five.  
The five emotions are happy, sad, angry, surprise, and neutral.  
Each of these five emotions has a different color, so each emotion can be more conveniently distinguished when a square is drawn around the face.  

#### The picture below is a diagram of how the code works overall.
![image](https://github.com/Iamchuchu/OpenSW_termProject/assets/144139251/e73372cd-2a45-4451-92dc-dfa116e71cd9)



#### Note
The code assumes the existence of certain utility functions and model files (like Haar Cascade file, emotion model file, etc.) that are not provided in the code snippet. Ensure that these dependencies are satisfied for the code to run successfully.

## Requirements: (with version i used)
- cv2 (4.8.1)
- tensorflow (2.15.0)
- scipy (1.11.4)
- opencv-python (4.8.1.78)
- keras (2.15.0)
- cvlib (0.2.7)
- numpy (1.26.2)
- pillow (10.1.0)
- pandas (2.1.3)
- matplotlib (3.8.2)
- h5py (3.10.0)
  - Models and Datasets Related Packages
    - emotion_model.hdf5
    - haarcascade_frontalface_default.xml
    - utils/datasets.py
    - utils/inference.py
    - utils/preprocessor
    
<https://github.com/petercunha/Emotion/tree/master>  
  
<https://bskyvision.com/entry/%EC%9B%B9%EC%BA%A0-%EC%98%81%EC%83%81-%EC%8B%A4%EC%8B%9C%EA%B0%84-%EB%AC%BC%EC%B2%B4-%EA%B2%80%EC%B6%9C-%ED%8C%8C%EC%9D%B4%EC%8D%AC>

You can download the Models and Datasets Related Packages you need here  
  
## Command to run  
    python real_final_code.py --image part 2/girl.jpg --width 0.955  
Before running this code, make sure that the file path, the package required, and the name of the image file are correct.  
If the above environment is not properly established, **it will not be implemented**.  
There is no problem even if you download the py file and run it on your IDE unless you've contracted the right coding environment.  

## Result  


![image](https://github.com/Iamchuchu/OpenSW_termProject/assets/144139251/d3871bf2-db74-4eac-b9c4-27aca4a3ff74)  
  
![image](https://github.com/Iamchuchu/OpenSW_termProject/assets/144139251/dbaad9cb-6bc0-4a78-9001-d8700e867581)  
  
![image](https://github.com/Iamchuchu/OpenSW_termProject/assets/144139251/7cd0660b-62c4-44d4-ab7f-f8b40d3ecae2)  

  
## The Limitation  
1. Due to the limitations of the database, only objects belonging to the data content can be recognized.  
  
2. The command of the code is drawn square with recognized subject attention, and if too many objects are recognized in one picture, it may be difficult for the user to grasp the given information.  

## Reference
Code Reference  
<https://github.com/petercunha/Emotion/tree/master>   
  
<https://bskyvision.com/entry/%EC%9B%B9%EC%BA%A0-%EC%98%81%EC%83%81-%EC%8B%A4%EC%8B%9C%EA%B0%84-%EB%AC%BC%EC%B2%B4-%EA%B2%80%EC%B6%9C-%ED%8C%8C%EC%9D%B4%EC%8D%AC>  
