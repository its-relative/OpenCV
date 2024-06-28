#!/usr/bin/env python
# coding: utf-8

# In[1]:


from deepface import DeepFace


# In[2]:


import cv2


# In[3]:


img = cv2.imread("image.jpg",1)


# In[4]:


cv2.imshow("Image", img)
cv2.waitKey(10000)
cv2.destroyAllWindows()


# In[5]:


prediction = DeepFace.analyze(img)


# In[6]:


prediction[0]["dominant_emotion"]


# ### Lets draw a rectangle around the face

# In[7]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# In[14]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1,4)


# In[15]:


for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0), 2)


# In[33]:


cv2.imshow("Image", img)
cv2.waitKey(100000)
cv2.destroyAllWindows()


# In[16]:


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, prediction[0]["dominant_emotion"],(50,100),font,3,(0,255,0),cv2.LINE_AA)


# In[ ]:


cv2.imshow("Face_detector",img)
cv2.waitKey(10000)
cv2.destroyAllWindows()

