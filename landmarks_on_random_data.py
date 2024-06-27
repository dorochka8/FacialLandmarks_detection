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
        'https://s1.r29static.com/bin/entry/23a/1080x1350,80/1536753/image.jpg',
        'https://s1.r29static.com/bin/entry/564/1080x1350,80/1536755/image.jpg',
        'https://s3.r29static.com/bin/entry/412/1080x1350,80/1536759/image.jpg',
        'https://cdn.facesofopensource.com/wp-content/uploads/2017/02/09202215/linus.faces22052.web_.jpg',
        'https://cdn.facesofopensource.com/wp-content/uploads/2022/05/10133823/KenThompson20516-1.web_-1229x1536.jpg',
        'https://cdn.facesofopensource.com/wp-content/uploads/2017/03/29182620/kristinepeterson.faces23463-2.web_.jpg',
        'https://cdn.facesofopensource.com/wp-content/uploads/2018/08/31025129/jessmckellar28843-1.jpg',
        'https://cdn.facesofopensource.com/wp-content/uploads/2017/03/21214903/stormy.faces23764.web_.jpg',
        'https://cdn.facesofopensource.com/wp-content/uploads/2016/04/23071145/faces.GuidovanRossum20593.web_.jpg',
        'https://cdn.facesofopensource.com/wp-content/uploads/2017/03/21210318/sallyk.faces23990.web_.jpg',
        'https://s1.r29static.com/bin/entry/43a/0,0,2000,2400/1440x1728,80/1536749/image.jpg',
        ]
imgs = download_imgs(URLs)

num_pic_to_show = 4
title = 'Examples of images'
show_examples(imgs, title, num_pic_to_show)

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

show_results(detector, predictor, imgs, num_samples_to_show=5)
