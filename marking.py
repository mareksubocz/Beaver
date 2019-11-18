import cv2 as cv
import numpy as np
from numpy.core.numeric import bitwise_not
from time import sleep

#Players


#pewnie trzeba będzie dodać pozycje linii siatki
def markPlayers(mask, courtcontours,img):
    contoursImage = img.copy()
    #mask = mask_teal + mask_orange
    mask = cv.dilate(mask, kernel, iterations=1)
    mask = cv.erode(mask, kernel, iterations=1)
    #Do masking
    res = cv.bitwise_and(img, img, mask=mask)
    #convert to hsv to gray
    res_bgr = cv.cvtColor(res,cv.COLOR_HSV2BGR)
    res_gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)


    #Defining a kernel to do morphological operation in threshold image to get better output
    kernel = np.ones((10,10),np.uint8)
    thresh = cv.threshold(res_gray,127,255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)


    #find contours in threshold image     
    contours,hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)


    font = cv.FONT_HERSHEY_SIMPLEX
    #Countour processing

    for c in contours:
            x,y,w,h = cv.boundingRect(c)

            #Check for contours within court
            if(cv.pointPolygonTest(courtcontours[0],(x+w/2, y+h/2),False)==-1):
                #Check for contours near court (distance is minus)
                if(cv.pointPolygonTest(courtcontours[0],(x+w/2, y+h/2),True)<=-30):
                    #print(cv.pointPolygonTest(contour,(x+w/2, y+h/2),True))
                    continue

            #Detect players
            if(h>=(1.1)*w):
                if(100>w>=15 and 100>h>= 25):
                    player_img = img[y:y+h,x:x+w]
                    player_hsv = cv.cvtColor(player_img,cv.COLOR_BGR2HSV)
                    ''' To jest to stare które wykrywa warunkiem ilości zgodnych pikseli, trza zastąpić stroną siatki
                    #First set
                    maskfirst = cv.inRange(player_hsv, np.array(lowerLimit[0]), np.array(upperLimit[0]))
                    res1 = cv.bitwise_and(player_img, player_img, mask=maskfirst)
                    res1 = cv.cvtColor(res1,cv.COLOR_HSV2BGR)
                    res1 = cv.cvtColor(res1,cv.COLOR_BGR2GRAY)
                    nzCountone = cv.countNonZero(res1)
                    #Second set
                    masksecond = cv.inRange(player_hsv, np.array(lowerLimit[1]), np.array(upperLimit[1]))
                    res2 = cv.bitwise_and(player_img, player_img, mask=masksecond)
                    res2 = cv.cvtColor(res2,cv.COLOR_HSV2BGR)
                    res2 = cv.cvtColor(res2,cv.COLOR_BGR2GRAY)
                    nzCounttwo = cv.countNonZero(res2)

                    if(nzCountone >= 2000):
                        #Mark first
                        cv.putText(contoursImage, 'Pierwsza', (x-2, y-2), font, 0.8, (255,0,0), 2, cv.LINE_AA)
                        cv.rectangle(contoursImage,(x,y),(x+w,y+h),(255,0,0),3)
                    else:
                        pass
                    if(nzCounttwo >= 2000):
                        #Mark second
                        cv.putText(contoursImage, 'Druga', (x-2, y-2), font, 0.8, (0,255,0), 2, cv.LINE_AA)
                        cv.rectangle(contoursImage,(x,y),(x+w,y+h),(0,255,0),3)
                    else:
                        pass
                    '''
                    cv.putText(contoursImage, 'Ktos', (x-2, y-2), font, 0.8, (0,255,0), 2, cv.LINE_AA)
                    cv.rectangle(contoursImage,(x,y),(x+w,y+h),(0,255,0),3)
    return contoursImage         

#End of markPlayers
#Ball marking
#Pewnie trzeba będzie dodać poprzednie lokalizacje piłki czy coś
def markBall(ballLowerLimit,ballUpperLimit,img,previmg):
    #Defining a kernel to do morphological operation in threshold image to get better output
    kernel = np.ones((5,5),np.uint8)
    contoursImage = img.copy()      
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)
    oldgray = cv.cvtColor(previmg, cv.COLOR_BGR2GRAY)
    oldgray = cv.GaussianBlur(oldgray, (21, 21), 0)
    delta = cv.absdiff(oldgray, gray)
    #maskball = cv.dilate(maskball, kernel, iterations=5)
    #maskball = cv.erode(maskball, kernel, iterations=5)
    #Do masking
    '''
    resball = cv.bitwise_and(img, img, mask=maskball)
    #convert to hsv to gray
    resball_bgr = cv.cvtColor(resball,cv.COLOR_HSV2BGR)
    resball_gray = cv.cvtColor(resball,cv.COLOR_BGR2GRAY)'''


   
    thresh = cv.threshold(delta,10,255,cv.THRESH_BINARY)[1]
    
    thresh = cv.dilate(thresh, kernel, iterations=5)
    thresh = cv.erode(thresh, kernel, iterations=3)
    #thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    font = cv.FONT_HERSHEY_SIMPLEX
    #find contours in threshold image     
    contoursss,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    nzCountball = []
    for c in contoursss:
            x,y,w,h = cv.boundingRect(c)
            #Check for ball
            if(2>w>=1 and 2>h>= 1):
                ball_img = img[y:y+h,x:x+w]
                ball_hsv = cv.cvtColor(ball_img,cv.COLOR_BGR2HSV)
                #white ball  detection
                maskball = cv.inRange(ball_hsv, np.array(ballLowerLimit), np.array(ballUpperLimit))
                res3 = cv.bitwise_and(ball_img, ball_img, mask=maskball)
                res3 = cv.cvtColor(res3,cv.COLOR_HSV2BGR)
                res3 = cv.cvtColor(res3,cv.COLOR_BGR2GRAY)
                nzCountball.append(cv.countNonZero(res3))
            else:
                 nzCountball.append(0)
    if len(nzCountball)!=0:
        x,y,w,h = cv.boundingRect(contoursss[nzCountball.index(max(nzCountball))])
        # show football
        cv.putText(contoursImage, 'Pilka', (x-2, y-2), font, 0.8, (0,255,0), 2, cv.LINE_AA)
        cv.rectangle(contoursImage,(x,y),(x+w,y+h),(0,255,0),3)
    return contoursImage
#end of showBall





img = cv.imread('field.jpg')
cap = cv.VideoCapture('cut.mp4')
# vidObj = cv.VideoCapture(0)
#Manual HSV limits for colors, first two are colors of shirts of both teams
lowerLimit= []
upperLimit= []
lowerLimit.append([0, 0, 171])
upperLimit.append([180, 70, 255])  

lowerLimit.append([1, 24, 18] )
upperLimit.append([148, 265, 84])
#Ball [16, 44, 149] [32, 105, 265]  [86, 19, 131] [97, 84, 246]
lowerLimit.append([15, 105, 64] )
upperLimit.append([30, 217, 131] )
#Background 
lowerLimit.append([50, 40, 40])
upperLimit.append([100, 255, 255])

lowerLimit.append([0, 60, 60])
upperLimit.append([15, 255, 255])



#Define a mask ranging from lower to uppper
mask = 0
#Combining masks
#lowerlimit[x] upperlimit[x]
#x = 0 || 1 => team colors
#x > 1 => background (hopefully won't be needed)
success, previmg = cap.read()
previmg = cv.resize(img, (800, 500))
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv.resize(img, (800, 500))
    # img = cv.bilateralFilter(img, 9, 75, 75)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Teal range [92, 121, 152] [100, 189, 221]
    lower_teal = np.array([50, 40, 40])
    upper_teal = np.array([100, 255, 255])

    # Orange range [3, 137, 244] [10, 174, 265]
    lower_orange = np.array([0, 120, 200])
    upper_orange = np.array([10, 200, 255])
    
    # Cream range [-3, 70, 243] [6, 96, 265]
    lower_cream = np.array([0, 70, 240])
    upper_cream = np.array([1, 100, 255])

    # Define a mask ranging from lower to uppper
    mask_orange = cv.inRange(hsv, lower_orange, upper_orange) + cv.inRange(hsv, lower_cream, upper_cream) 
    mask_1 = mask_orange.copy()
    mask_teal = cv.inRange(hsv, lower_teal, upper_teal)
    
    contoursImage = img.copy()
    # caly parkiet
    # mask_court = cv.bitwise_or(mask_teal, mask_orange)

    # smoothing
    '''
    kernel = np.ones((5, 5), np.uint8)
    mask_orange = cv.dilate(mask_orange, kernel, iterations=5)
    mask_orange = cv.erode(mask_orange, kernel, iterations=5)
    # cv.imshow('mask', mask_orange)

    
    contours, hierarchy = cv.findContours(
        mask_orange.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours_areas = list(map(cv.contourArea, contours))
    # largest contour is court contour
    if contours_areas:
        court_contour = [contours[contours_areas.index(max(contours_areas))]]

        court_mask = np.zeros((500, 800))
        cv.fillPoly(court_mask, pts=court_contour, color=(1))

        court_mask = np.array(court_mask, dtype=np.uint8)
        court_mask *= 255

        canny_court_mask = cv.Canny(court_mask, 50, 150)
        # cv.imshow('canny_court_mask', canny_court_mask)

        contours, hierarchy = cv.findContours(
            court_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        hull = []
        for contour in contours:
            hull.append(cv.convexHull(contour, False))

        if cv.contourArea(hull[0]) > 20000:
            cv.drawContours(contoursImage, hull, 0,
                            (0, 0, 255), thickness=4)
    '''
    maskball = cv.inRange(hsv, np.array(lowerLimit[2]), np.array(upperLimit[2]))
    contoursImage = markBall(lowerLimit[2],upperLimit[2],img,previmg)
    
    
    #Show results
    previmg = img.copy()
    cv.imshow('1', contoursImage)
    cv.moveWindow('1', 200, 100)
    sleep(0.03)
    # cv.waitKey()
    # cv.namedWindow("1", cv.WINDOW_AUTOSIZE)
    # cv.destroyAllWindows()
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()