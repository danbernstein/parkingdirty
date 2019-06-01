import os

if not os.path.exists('object_detection/input_imgs'):
    os.makedirs('object_detection/input_imgs')
    
    # download parking dirty images here when needed

if not os.path.exists('object_detection/output_imgs'):
    os.makedirs('object_detection/output_imgs')

if not os.path.exists('object_detection/output_csv'):
    os.makedirs('object_detection/output_csv')
    
    
if not os.path.exists('object_detection/output_xml'):
    os.makedirs('object_detection/output_xml/xml_files')
