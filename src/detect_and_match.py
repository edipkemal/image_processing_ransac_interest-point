# Edip Kemal Sardogan
# 240201026
import cv2 as cv

image_number=6

images=[]

path="../data/goldengate/goldengate-0{}.png"
siftpath="../data/goldengate/sift_keypoints_{}.png"
tentativepath="../data/goldengate/tentative_correspondences_{}-{}.png"
sift= cv.SIFT_create()

for i in range(image_number):
    img = cv.imread(path.format(i))
    images.append(img)
    kp = sift.detect(img,None)
    simg = cv.drawKeypoints(img,kp,None)
    cv.imwrite(siftpath.format(i),simg)
    

kp_des=[]
for i in range(image_number):
    img=images[i]
    kp, des=sift.detectAndCompute(img,None)
    kp_des.append((kp,des))
    kp2, des2=kp_des[i]
    f=open("../data/goldengate/sift_{}.txt".format(i),"w")
    for k in kp2:
        f.write("{},{}\n".format(k.pt[0],k.pt[1]))
    f.write("descriptors\n")
    for d in des2:
        f.write("{}\n".format(d))
    f.close()
    if i != 0:
        img1=images[i-1]
        img2=images[i]
        
        kp1, des1=kp_des[i-1]
        

        bf=cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches=bf.knnMatch(des1,des2,k=2)
    
        f=open("../data/goldengate/tentative_correspondences_{}-{}.txt".format(i-1,i),"w")
        match = []
        for m,n in matches:
            if m.distance < 0.6*n.distance:
                match.append([m])
                f.write("{},{}\n".format(m.queryIdx, m.trainIdx))
        
        f.close()
        
            
        img3=cv.drawMatchesKnn(img1,kp1,img2,kp2,match,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
        cv.imwrite(tentativepath.format(i-1,i),img3)   
