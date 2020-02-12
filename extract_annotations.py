import os
import shutil


# RUN ONLY ONE TIME TO GET XML FILES !!!

def copy_xml_labels(images_dir, labels_dir):
    labels_dir_files = os.listdir(labels_dir)
    if len(labels_dir_files) == 0:
        for img in os.listdir(images_dir):
            img_name = os.path.splitext(img)[0]
            xml_name = img_name + '.xml'
            old_file = os.path.join(annotations_dir, xml_name)
            new_file = os.path.join(labels_dir, xml_name)
            shutil.copyfile(old_file, new_file)
        print('-------------------------------------------------------------------')
        print(' Done')
        print('-------------------------------------------------------------------')
    else:
        print('-------------------------------------------------------------------')
        print(' XML files already copied to ' + labels_dir)
        print(' Num. of items: ' + str(len(labels_dir_files)))
        print('-------------------------------------------------------------------')


annotations_dir = 'datasets/BelgianLicencePlates/AnnotationsXML/001plate'

test_img_dir = 'datasets/BelgianLicencePlates/TestPlates'
test_labels_dir = 'datasets/BelgianLicencePlates/TestPlatesLabels'

copy_xml_labels(test_img_dir, test_labels_dir)
