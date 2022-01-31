import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

# assign teeth to quadrant
def quadrant(df, img):
    # get image shape
    img_height = img.shape[0]
    img_width = img.shape[1]
    # initiate quadrant column
    df['quadrant'] = ''
    # for each tooth calculate position in x-ray and assign quadrant
    for i in range(df.shape[0]):
        if df['xmin'][i] < img_width/2:
            if df['ymin'][i] < img_height/2:
                df['quadrant'][i]=1
            else: df['quadrant'][i]=4
        else:
            if df['ymin'][i] < img_height/2:
                df['quadrant'][i]=2
            else: df['quadrant'][i]=3

    # sort values by quadrant and left to right        
    df.sort_values(by=['quadrant', 'xmin'], inplace=True, ignore_index=True)
    # assign teeth to left/right
    average_x = df['xmin'].mean()
    df['left/right'] = np.where(df.xmin < average_x, 'left', 'right')
    # check if in wrong quadrant horizontally
    for i in df.index:
        if (df.loc[i, 'quadrant']==1) & (df.loc[i, 'left/right']=='right'):
            df.loc[i, 'quadrant'] = 2
        elif (df.loc[i, 'quadrant']==4) & (df.loc[i, 'left/right']=='right'):
            df.loc[i, 'quadrant'] = 3
        elif (df.loc[i, 'quadrant']==2) & (df.loc[i, 'left/right']=='left'):
            df.loc[i, 'quadrant'] = 1
        elif (df.loc[i, 'quadrant']==3) & (df.loc[i, 'left/right']=='left'):
            df.loc[i, 'quadrant'] = 4
    # drop left/right column
    df.drop(columns=['left/right'], inplace=True)
    # assign teeth to top/bottom
    average_y = df['ymin'].mean()
    df['top/bottom'] = np.where(df.ymin < average_y, 'top', 'bottom')
    # check if in wrong quadrant vertically
    for i in df.index:
        if (df.loc[i, 'quadrant']==1) & (df.loc[i, 'top/bottom']=='bottom'):
            df.loc[i, 'quadrant'] = 4
        elif (df.loc[i, 'quadrant']==2) & (df.loc[i, 'top/bottom']=='bottom'):
            df.loc[i, 'quadrant'] = 3
        elif (df.loc[i, 'quadrant']==3) & (df.loc[i, 'top/bottom']=='top'):
            df.loc[i, 'quadrant'] = 2
        elif (df.loc[i, 'quadrant']==4) & (df.loc[i, 'top/bottom']=='top'):
            df.loc[i, 'quadrant'] = 1
    # drop top/bottom column
    df.drop(columns=['top/bottom'], inplace=True)
    # sort values by quadrant and left to right   
    df.sort_values(by=['quadrant', 'xmin'], inplace=True, ignore_index=True)
    return df

# creates a dictionary broken up by quadrant
def quad_dict(data):
    q1_data = data[data['quadrant']==1]
    q1_data.sort_values(by='xmin', inplace=True)
    q1_data.reset_index(drop=True, inplace=True)

    q2_data = data[data['quadrant']==2]
    q2_data.reset_index(drop=True, inplace=True)

    q3_data = data[data['quadrant']==3]
    q3_data.reset_index(drop=True, inplace=True)

    q4_data = data[data['quadrant']==4]
    q4_data.sort_values(by='xmin', inplace=True)
    q4_data.reset_index(drop=True, inplace=True)

    quadrant_dict = {'1': q1_data, '2': q2_data, '3': q3_data, '4': q4_data}
    return(quadrant_dict)

# get dataframe of missing teeth and append to df
def find_missing_teeth(df, quadrant_dict):
    # add missing teeth column to df
    df['missing teeth'] = 0
    # initate list of missing teeth
    missing_teeth = []
    # loop through quadrant to find missing teeth
    for quadrant in quadrant_dict.keys():
        if len(quadrant_dict[quadrant]) < 8:
            for i in range(len(quadrant_dict[quadrant])-1):
                tooth_dist = quadrant_dict[quadrant]['xmin'][i+1] - quadrant_dict[quadrant]['xmax'][i]
                if tooth_dist > 0.005: # average_dist
                    xmin = quadrant_dict[quadrant]['xmin'][i+1]
                    ymin = (quadrant_dict[quadrant]['ymin'][i+1] + quadrant_dict[quadrant]['ymin'][i])/2
                    xmax = quadrant_dict[quadrant]['xmax'][i]
                    ymax = (quadrant_dict[quadrant]['ymax'][i+1] + quadrant_dict[quadrant]['ymax'][i])/2
                    _class = 'missing'
                    missing = 1
                    missing_teeth.append([xmin, ymin, xmax, ymax, _class, int(quadrant), missing])
    missing_teeth = pd.DataFrame(missing_teeth, 
                                 columns = ['xmin', 'ymin', 'xmax', 'ymax', 'class', 'quadrant', 'missing teeth'])
    df = df.append(missing_teeth)
    df.sort_values(by=['quadrant','xmin'], inplace=True, ignore_index=True)
    return df

# number teeth according to ISO
def number_ISO(df):
    # calculate for quadrant 1
    q1_data = df[df['quadrant']==1]
    q1_data.sort_values(by='xmin', ascending=False, inplace=True, ignore_index=True)
    q1_data['ISO_tooth_label'] = ''
    for i in range(q1_data.shape[0]):
        q1_data.iloc[i, q1_data.columns.get_loc('ISO_tooth_label')] = '1'+str(i+1)
    # calculate for quadrant 2
    q2_data = df[df['quadrant']==2]
    q2_data.sort_values(by='xmin', inplace=True, ignore_index=True)
    q2_data['ISO_tooth_label'] = ''
    for i in range(q2_data.shape[0]):
        q2_data.iloc[i, q2_data.columns.get_loc('ISO_tooth_label')] = '2'+str(i+1)
    # calculate for quadrant 3
    q3_data = df[df['quadrant']==3]
    q3_data.sort_values(by='xmin', inplace=True, ignore_index=True)
    q3_data['ISO_tooth_label'] = ''
    for i in range(q3_data.shape[0]):
        q3_data.iloc[i, q3_data.columns.get_loc('ISO_tooth_label')] = '3'+str(i+1)
    # calculate for quadrant 4
    q4_data = df[df['quadrant']==4]
    q4_data.sort_values(by='xmin', ascending=False, inplace=True, ignore_index=True)
    q4_data['ISO_tooth_label'] = ''
    for i in range(q4_data.shape[0]):
        q4_data.iloc[i, q4_data.columns.get_loc('ISO_tooth_label')] = '4'+str(i+1)
    # combine quadrant dataframes together
    df_final = pd.concat([q1_data, q2_data, q3_data, q4_data], ignore_index=True)
    return df_final

# map labels
def label_map(df):
    # define classes
    class_dict = {0: 'tooth', 1: 'test'}
    df.replace({"class": class_dict}, inplace=True)
    return df

# label image with bounding boxes
def label_image(image, df):
    dummy = image.copy()
    for i in range(df.shape[0]):
        # set colour of bounding boxes
        if df['missing teeth'][i] == 0:
            color = (255, 0, 0)
        else: 
            color = (0, 255, 255)
        # set edges of bounding boxes
        start = (int(df['xmin'][i]), int(df['ymin'][i]))
        end = (int(df['xmax'][i]), int(df['ymax'][i]))
        # set the text location based on quadrant
        if (df['quadrant'][i] == 1) | (df['quadrant'][i] == 2):
            label = (start[0], start[1]-25)
        else: 
            label = (start[0], end[1]+45)
        # draw the rectangles on the image
        cv2.rectangle(dummy, start, end, color, 3)
        # add the labels to the image   
        text = str(df['ISO_tooth_label'][i]) + ':' + str(df['class'][i])
        cv2.putText(dummy, text, label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # save image  
    cv2.imwrite('static/image1.jpg', dummy)
    
# main function to run all the code
def main(df, image):
    # get quadrant column
    df = quadrant(df, image)
    # create a dictionary by quadrant
    quadrant_dict = quad_dict(df)
    # find missing tooth data and add to df
    df = find_missing_teeth(df, quadrant_dict)
    # number teeth
    df = number_ISO(df)
    # map labels to tooth type
    df = label_map(df)
    # label and save the image
    label_image(image, df)