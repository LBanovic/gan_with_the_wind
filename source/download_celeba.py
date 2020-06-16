from mega import Mega

import sys

resolution = int(sys.argv[1])
mega = Mega()
m = mega.login('luka.banovich@gmail.com', 'd!W3P2!B2G')
if len(sys.argv) == 2:
    for res in range(2, resolution+1):
        tfrecord = f'tfrecords-r0{res}.tfrecords'
        file = m.find(tfrecord)
        print(f'Downloading {tfrecord}')
        m.download(file)
else:
    tfrecord = f'tfrecords-r0{resolution}.tfrecords'
    file = m.find(tfrecord)
    print(f'Downloading {tfrecord}')
    m.download(file)