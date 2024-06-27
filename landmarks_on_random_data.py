import dlib

from main_functions import download_imgs, show_examples, show_results

URLs = ['https://s1.r29static.com/bin/entry/444/1080x1350,80/1536751/image.jpg',
        'https://s1.r29static.com/bin/entry/df7/1253x1766,80/1536764/image.jpg',
        'https://s1.r29static.com/bin/entry/d35/1080x1350,80/1536752/image.jpg',
        'https://s2.r29static.com/bin/entry/39f/1080x1350,80/1536760/image.jpg',
        'https://s1.r29static.com/bin/entry/49b/1080x1350,80/1536757/image.jpg',
        'https://s2.r29static.com/bin/entry/839/1080x1350,80/1536765/image.jpg',
        'https://s1.r29static.com/bin/entry/dc1/1080x1350,80/1536756/image.jpg',
        'https://s2.r29static.com/bin/entry/18f/1080x1350,80/1536758/image.jpg',
        'https://s2.r29static.com/bin/entry/9ac/1080x1350,80/1536761/image.jpg',
        'https://s2.r29static.com/bin/entry/cd4/1080x1350,80/1536763/image.jpg',
        ]
imgs = download_imgs(URLs)

num_pic_to_show = 4
title = 'Examples of images'
show_examples(imgs, title, num_pic_to_show)

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

show_results(detector, predictor, imgs, num_samples_to_show=5)