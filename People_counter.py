import cv2
import imutils

total=0
state=0

count=0
def detect(frame):
    global total
    global state
    
    global count
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame)#, winStride = (8, 8))#, padding = (8, 8), scale = 1.0)
    
    
    person = 1
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
       
    cv2.putText(frame, f'Total current : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    if state == 0:
        if person > 2:
            total += person-1
            state=1
            
    if state == 1 :
        if person <= 1:
            state=0
    cv2.putText(frame, f'Total Pedestrians : {int(total/2)}', (40,100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)
    print(state)
    return frame

def detectByPathVideo(path):#, writer):
    global count
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found')
        return

    print('Detecting people...')
    while video.isOpened():
        check, frame =  video.read()

        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            frame = detect(frame)
            
            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()


def humanDetector():
    video_path = 'Video_test3.mp4'

    if video_path is not None:
        print('[INFO] Opening Video .')
        detectByPathVideo(video_path)#, writer)

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    humanDetector()



