import pandas as pd
import sys
import math

def distance(coordinate1,coordinate2):
    return math.sqrt(pow(abs(coordinate1[0] - coordinate2[0]),2) + pow(abs(coordinate1[1]-coordinate2[1]),2))

def get_iou(dict_python,dict_cpp):
    x_left = max(dict_python['x1'] ,dict_cpp['x1'])
    y_top = max(dict_python['y1'] ,dict_cpp['y1'])
    x_right = min(dict_python['x2'] ,dict_cpp['x2'])
    y_bottom = min(dict_python['y2'] ,dict_cpp['y2'])

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (dict_python['x2'] - dict_python['x1']) * (dict_python['y2'] - dict_python['y1'])
    bb2_area = (dict_cpp['x2'] - dict_cpp['x1']) * (dict_cpp['y2'] - dict_cpp['y1'])

    iou = intersection_area/ float(bb1_area + bb2_area - intersection_area)
    
    return iou

csvData = pd.read_csv("baboons_python.csv")
frame_number_python = sys.argv[1]
csvData = csvData.loc[csvData['frame']==int(frame_number_python)]
new_csvData = csvData.sort_values(by='x1')
new_path_python="baboons_python_sorted.csv"
#new_csvData.to_csv(new_path_python,index=False)
num_elements_python = new_csvData.shape[0]
csvData_python = new_csvData

csvData = pd.read_csv("baboons_c++.csv")
frame_number = sys.argv[1]
csvData = csvData.loc[csvData['frame']==int(frame_number)]
new_csvData = csvData.sort_values(by='x1')
num_elements_cpp = new_csvData.shape[0]
if num_elements_cpp > num_elements_python:
    new_csvData = new_csvData.iloc[:num_elements_cpp]
new_path_cpp = "baboons_c++_sorted.csv"
#new_csvData.to_csv(new_path,index=False)
csvData_cpp = new_csvData


if num_elements_cpp > num_elements_python:
    csvData_cpp = csvData_cpp.iloc[:num_elements_python]
else:
    csvData_python = csvData_python.iloc[:num_elements_cpp]


csvData_python.to_csv(new_path_python,index=False)
csvData_cpp.to_csv(new_path_cpp,index=False)

csvData_cpp = pd.read_csv("baboons_c++_sorted.csv")
tl_cpp = []
br_cpp = []


for index,row in csvData_cpp.iterrows():
    x1 = row['x1']
    y1 = row['y1']
    x2 = row['x2']
    y2 = row['y2']
    tl = (x1,y1)
    tl_cpp.append(tl)
    br = (x2,y2)
    br_cpp.append(br)


print(tl_cpp)
csvData_python = pd.read_csv("baboons_python_sorted.csv")
tl_python = []
br_python = []


for index,row in csvData_python.iterrows():
    x1 = row['x1']
    y1 = row['y1']
    x2 = row['x2']
    y2 = row['y2']
    tl = (x1,y1)
    br = (x2,y2)
    tl_python.append(tl)
    br_python.append(br)
print(tl_python)
 
for index_cpp,index_python in zip(tl_cpp,tl_python):
    x1_coordinate_cpp = index_cpp[0]
    x1_coordinate_python=index_python[0]
    print(x1_coordinate_cpp,x1_coordinate_python)

list_of_coordinates_for_comparison = []
python_coordinates_list = []
cpp_coordinates_list = []
for python_tuple_tl,python_tuple_br in zip(tl_python,br_python):   
    for cpp_tuple_tl,cpp_tuple_br in zip(tl_cpp,br_cpp):
        x1_coordinate_python = python_tuple_tl[0]
        y1_coordinate_python = python_tuple_tl[1]
        nearest_tl = min(tl_cpp,key=lambda x:distance(x,python_tuple_tl))
        x1_coordinate_delta = x1_coordinate_python - nearest_tl[0]
        y1_coordinate_delta = y1_coordinate_python - nearest_tl[1]
        x2_coordinate_python = python_tuple_br[0]
        y2_coordinate_python = python_tuple_br[1]
        nearest_br = min(br_cpp,key=lambda x:distance(x,python_tuple_br))
        x2_coordinate_delta = x2_coordinate_python - nearest_br[0]
        y2_coordinate_delta = y2_coordinate_python - nearest_br[1]
        if ((x1_coordinate_delta <15) and (y1_coordinate_delta<15) and (x2_coordinate_delta <15) and (y2_coordinate_delta < 15)):
            #python_coordinate = [(x1_coordinate_python,y1_coordinate_python),(x2_coordinate_python,y2_coordinate_python)]
            #cpp_coordinate = [(nearest_tl[0],nearest_tl[1]),(nearest_br[0],nearest_br[1])]
            python_coordinate = {'x1':x1_coordinate_python,'x2':x2_coordinate_python,'y1':y1_coordinate_python,'y2':y2_coordinate_python}
            cpp_coordinate = {'x1':nearest_tl[0],'x2':nearest_br[0],'y1':nearest_tl[1],'y2':nearest_br[1]}
            python_coordinates_list.append(python_coordinate)
            cpp_coordinates_list.append(cpp_coordinate)
            #list_of_coordinates_for_comparison.append(coordinates)
            break


print(python_coordinates_list)
print(cpp_coordinates_list)

print("The number of elemets are")
print (len(python_coordinates_list),len(cpp_coordinates_list),num_elements_cpp,num_elements_python)
sum_iou_element = 0
for dict_item_cpp,dict_item_python in zip(cpp_coordinates_list,python_coordinates_list):
    iou_element = get_iou(dict_item_python,dict_item_cpp)
    sum_iou_element += iou_element
    print(iou_element)

average_iou = sum_iou_element/len(cpp_coordinates_list)
print ("Average is =")
print(average_iou)
#print(list_of_coordinates_for_comparison)
#print(len(list_of_coordinates_for_comparison))
def distance(coordinate1,coordinate2):
    return sqrt(pow(abs(coordinate1[0] - coordinate2[0],2) + pow(abs(coordinate1[1]-coordinate2[1],2))))
