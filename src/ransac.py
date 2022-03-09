# Edip Kemal Sardogan
# 240201026
import numpy as np
import cv2 as cv

image_number=6
path="../data/goldengate/goldengate-0{}.png"
homography_path="../data/goldengate/h_{}-{}.txt"
inl_img_path="../data/goldengate/inliers_{}-{}.png"
inl_path="../data/goldengate/inliers_{}-{}.txt"
for i in range(image_number-1):
    img1 = cv.imread(path.format(i),0) 
    img2 = cv.imread(path.format(i+1),0)
    img11 = cv.imread(path.format(i)) 
    img22= cv.imread(path.format(i+1))
    sift = cv.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 1
    
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    f=open(inl_path.format(i,i+1),"w")
    match = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            match.append(m)
            f.write("{},{}\n".format(m.queryIdx, m.trainIdx))
    
    f.close()
            
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in match ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in match ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

    f=open(homography_path.format(i,i+1),"w")
    for a in M:
        f.write("{}\n".format(a))

    f.close()
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,match,None,**draw_params)
    cv.imwrite(inl_img_path.format(i,i+1),img3)   
