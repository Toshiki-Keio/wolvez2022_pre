import cv2

def pickup_frame():
    mv=cv2.VideoCapture("img_data/exp_mov2.mov")
    frame_rate = int(mv.get(cv2.CAP_PROP_FPS))
    frame_count = int(mv.get(cv2.CAP_PROP_FRAME_COUNT))

    size=(mv.get(cv2.CAP_PROP_FRAME_WIDTH),mv.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(size)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 保存形式
    save = cv2.VideoWriter('img_data/exp_mov2_extracted.mov',
                        fourcc, frame_rate, (int(size[0]),int(size[1])))

    for i in range(0, frame_count-1, 2):
        ch, frame = mv.read()  # 2フレームずつ取り出す
        if ch:
            print(frame.shape)
            #for mov1
            #canvas=frame[38:int(mv.get(cv2.CAP_PROP_FRAME_HEIGHT)),58:int(mv.get(cv2.CAP_PROP_FRAME_WIDTH)),:]
            #for mov2
            canvas=frame[35:int(mv.get(cv2.CAP_PROP_FRAME_HEIGHT)),6:int(mv.get(cv2.CAP_PROP_FRAME_WIDTH)),:]
            #cv2.circle(frame,(6,35),2,(0,0,255))
            canvas=cv2.resize(canvas,(int(size[0]),int(size[1])))
            save.write(canvas)
    save.release()
    mv.release()

pickup_frame()