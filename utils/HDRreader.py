import cv2
import numpy as np
import Imath, OpenEXR

def rgbe2float(rgbe):
    res = np.zeros((rgbe.shape[0], rgbe.shape[1], 3))
    p = rgbe[:,:,3] > 0
    m = 2.0**(rgbe[:,:,3][p]-136.0)
    res[:,:,0][p] = rgbe[:,:,0][p] * m 
    res[:,:,1][p] = rgbe[:,:,1][p] * m 
    res[:,:,2][p] = rgbe[:,:,2][p] * m
    return res
    
    
def readHDR(fileName):
    fileinfo = {}
    with open(fileName, 'rb') as fd:
        tline = fd.readline().strip()
        # print((tline))

        if len(tline)<3 or tline[:2] != b'#?':
            # print ('invalid header')  
            return
        fileinfo['identifier'] = tline[2:]
 
        tline = fd.readline().strip()
        while tline:
            n = tline.find(b'=')
            if n>0:
                fileinfo[tline[:n].strip()] = tline[n+1:].strip()
            tline = fd.readline().strip()
 
        tline = fd.readline().strip().split(b' ')
        # print(tline)
        fileinfo['Ysign'] = tline[0][0]
        fileinfo['height'] = int(tline[1])
        fileinfo['Xsign'] = tline[2][0]
        fileinfo['width'] = int(tline[3])
        
        d = fd.read(1)
        data = []
        while d:
            data.append(ord(d))
            d = fd.read(1)
        # data = [ord(d) for d in fd.read()]
        height, width = fileinfo['height'], fileinfo['width']

        # print(len(data))

        img = np.zeros((height, width, 4))
        dp = 0
        for h in range(height):
            if data[dp] !=2 or data[dp+1]!=2:
                print ('this file is not run length encoded')
                print (data[dp:dp+4])
                return
            if data[dp+2]*256+ data[dp+3] != width:
                print ('wrong scanline width')
                return
            dp += 4
            for i in range(4):
                ptr = 0
                while(ptr < width):
                    if data[dp]>128:
                        count = data[dp]-128
                        if count==0 or count>width-ptr:
                            print ('bad scanline data')
                        img[h, ptr:ptr+count,i] = data[dp+1]
                        ptr += count
                        dp += 2
                    else:
                        count = data[dp]
                        dp += 1
                        if count==0 or count>width-ptr:
                            print ('bad scanline data')
                        img[h, ptr:ptr+count,i] = data[dp: dp+count]
                        ptr += count
                        dp +=count
    return img


def rbg2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def readEXR(hdrfile):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    hdr_t = OpenEXR.InputFile(hdrfile)
    dw = hdr_t.header()['dataWindow']
    size = (dw.max.x-dw.min.x+1, dw.max.y-dw.min.y+1)
    rstr = hdr_t.channel('R', pt)
    gstr = hdr_t.channel('G', pt)
    bstr = hdr_t.channel('B', pt)
    r = np.frombuffer(rstr, dtype=np.float32)
    r.shape = (size[1], size[0])
    g = np.frombuffer(gstr, dtype=np.float32)
    g.shape = (size[1], size[0])
    b = np.frombuffer(bstr, dtype=np.float32)
    b.shape = (size[1], size[0])
    res = np.stack([r,g,b], axis=-1)
    imhdr = np.asarray(res)
    return imhdr

def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        R = (img[:,:,0]).astype(np.float16).tostring()
        G = (img[:,:,1]).astype(np.float16).tostring()
        B = (img[:,:,2]).astype(np.float16).tostring()
        out.writePixels({'R' : R, 'G' : G, 'B' : B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)
