import os
from datetime import datetime
import cv2
import numpy as np 
import glob
import sys




def getDarkFrame(temp):
    darkpath=''# put path to dark file here
    png_files = glob.glob(os.path.join(darkpath, '*.png'))

    closest_dark_temp=1000000#some large value   
    closest_dark_temp_file=''
    for file in png_files:
        filename=file.split('/')[9]
        darktemp=int(filename.split('_')[5].split('.')[0])

        val=abs(temp-darktemp)
        if val<closest_dark_temp:
            closest_dark_temp=val
            closest_dark_temp_file=file
        #print(filename)
        #print(darktemp)
        #print("val: "+str(val))


    #print('closest dark temp file:')
    #print(closest_dark_temp_file)   
    #print('temp: '+str(temp))

    return closest_dark_temp_file


def motiondetection(newFramename,prevFramename):
    #do algnement and cropping before subtraction
    #print("whyyyyyyyy")
    #print("new frame name" +newFramename)
    image_new=cv2.imread(newFramename)
    image_prev=cv2.imread(prevFramename)

    #cv2.imshow('llll', image_prev)
    #cv2.waitKey(0)

    image_new,image_prev=circleMask(image_new,image_prev)
    #image_new,image_prev=circleCrop(image_new,image_prev)

    #8 for orignal path
    #df_file_new= cv2.imread(getDarkFrame(int(newFramename.split('/')[8].split('_')[4])))
    #df_file_prev= cv2.imread(getDarkFrame(int(prevFramename.split('/')[8].split('_')[4])))


    df_file_new= cv2.imread(getDarkFrame(int(newFramename.split('/')[7].split('_')[4])))
    df_file_prev= cv2.imread(getDarkFrame(int(prevFramename.split('/')[7].split('_')[4])))

    #cv2.imshow('df_file_new', df_file_new)
    #cv2.imshow('df_file_prev', df_file_prev)
    #cv2.imshow('Motion-Sub', subtracted_image_motion)
    #cv2.waitKey(0)

    #subtract dark frame 
    subtracted_image_new = cv2.subtract(image_new, df_file_new)#,dtype=cv2.CV_64F)
    #subtracted_image_prev=cv2.subtract(image_prev, df_file_prev)#,dtype=cv2.CV_64F)

    #subtract previous from new frame
    #subtracted_image_motion = cv2.subtract(subtracted_image_new, subtracted_image_prev)#,dtype=cv2.CV_64F)
    subtracted_image_motion = cv2.subtract(image_new, image_prev)#,dtype=cv2.CV_64F)

    #cv2.imshow('New image original', image_new)
    #cv2.imshow('New image dark subtract', subtracted_image_new)
    #cv2.imshow('Motion-Sub', subtracted_image_motion)
    #cv2.waitKey(0)

    return subtracted_image_motion,subtracted_image_new
    #cv2.imshow('Sub_new', subtracted_image_new)
    #cv2.imshow('Sub_prev', subtracted_image_prev)



    
    #cv2.imshow('Original Image', original_photo)
    #cv2.imshow('Motion-Sub', subtracted_image_motion)
    #cv2.waitKey(0)


def circleMask(newImage,prevImage):


    #np.zeros(newImage.shape())
    mask=np.zeros_like(newImage)
    #prevmask=np.zeros_like(prevImage)

    center=(640,480)
    radius=580

    cv2.circle(mask, center, radius, (255, 255, 255), -1)  # -1 indicates to fill 
    
    # Apply the mask to the original image
    new_masked_image = cv2.bitwise_and(newImage, mask)
    prev_masked_image = cv2.bitwise_and(prevImage, mask)


    #cv2.imshow('Cirlce Mask', mask)
    #cv2.imshow('New Cirlce Mask', new_masked_image)
    #cv2.imshow('Prev Cirlce Mask', prev_masked_image)
    #cv2.waitKey(0)

    return new_masked_image, prev_masked_image


def linedetection(image,og_image,newFramename):

    motion_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(motion_gray, 25, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(binary_mask, 50, 150, apertureSize=3)

    #cv2.imshow('edges', edges)
    #cv2.waitKey(0)
    #lines = cv2.HoughLines(edges, 1, np.pi/180, 20)


    # Apply HoughLinesP method to 
    # to directly obtain line end points
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=10, # Min number of votes for valid line
                minLineLength=5, # Min allowed length of line
                maxLineGap=5 # Max allowed gap between line for joining them
                )
    
    # Iterate over points
    if lines is None:  #check if zero lines are found in image
        
        print("lines is nonetype")
        return



    path=''#path where output of houghline algorithm will be
    name=newFramename.split('/')[7]
    folder=newFramename.split('/')[6]


    if os.path.isdir(path+folder):
        #return
        pass
    else:
        os.mkdir(path+folder)


    
    

    linecount=0
    print( "number of lines"+str(len(lines)))
    
    
    
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        # Draw the lines joing the points
        # On the original image
        #cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

        print(str(points))
        #og_image=cv2.rectangle(og_image, (x1-40,y1-40) , (x2+40,y2+40), (255, 0, 0) , 2)    
        #og_image=cv2.rectangle(og_image, (x1,y1) , (x2,y2), (255, 0, 0) , 2)    
        

        d=dirpath.split('/')[len(newFramename.split('/'))-2]

        if os.path.isdir(path+d):
            pass
        else:
            os.mkdir(path+d)


        og_image=cv2.rectangle(og_image, (x1-40,y1-40) , (x2+40,y2+40), (0, 255, 255) , 2)    

        try:
            cropped_image = image[y1-40:y2+40, x1-40:x2+40]      
            linecount+=1
            cv2.imwrite(path+d+"/"+name+"_line_"+str(linecount)+".png",cropped_image)
            print("attempted to save image: "+name)
        except Exception as e:
            print(e)
            continue

        if linecount>=8:
            break


    cv2.imwrite(path+folder+"/"+name,og_image)
        # Maintain a simples lookup list for points
        #lines_list.append([(x1,y1),(x2,y2)])
        

    #color =    
    
    # Save the result image
    #cv2.imshow('detectedlines',og_image)
    #cv2.waitKey(0)
    #cv2.imwrite('detectedLines.png',image)
        #cv2.imshow('Motion Binary', binary_mask)



# Specify the root directory


root_dir = ''#path of root foolder 


# Iterate over the root directory and its subdirectories
prev_file_path=""
prev_dirpath=""
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:  
        #print ("file type: "+ file[-4:])
        if file[-4:]==".png" and dirpath.split('/')[len(dirpath.split('/'))-1].startswith('202402') :

            


            #print ("file name: "+ file)
            file_path = os.path.join(dirpath, file)
            print("file path: "+file_path)
            # Get the age of the file in days
            #file_creation_time = os.path.getctime(file_path)
            #age_in_days = (datetime.now() - datetime.fromtimestamp(file_creation_time)).days
            # Process the file based on its age
            # (e.g., perform specific actions on files older than a certain age)
            #print(prev_file+"    "+file)
            
            if prev_dirpath!=dirpath:
                prev_file_path==""

            if prev_file_path=="":
                prev_file_path=file_path
                continue
            else:
                #new_frame_name=file_path
                #prev_frame_name=prev_file_path

                print("prev file path "+prev_file_path)
                motion_image,sub_image_new=motiondetection(file_path,prev_file_path)
                linedetection(motion_image,sub_image_new,file_path)


                #print(prev_frame_name+"    "+new_frame_name)
                prev_file_path=file_path
            prev_dirpath=dirpath    
            