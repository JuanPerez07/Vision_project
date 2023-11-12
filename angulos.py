def calcular_angulo(a,b,c):
    a=np.array(a) #Hombro
    b=np.array(b) #Codo
    c=np.array(c) #Muñeca

    radianes=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angulo=np.abs(radianes*180.0/np.pi)

    if angulo>180.0:
        angulo=360-angulo
    
    return angulo

hombro=[landmarks[mp_pose.PoseLandmarks.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmarks.LEFT_SHOULDER.value].y]
codo=[landmarks[mp_pose.PoseLandmarks.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmarks.LEFT_ELBOW.value].y]
muñeca=[landmarks[mp_pose.PoseLandmarks.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmarks.LEFT_WRIST.value].y]

calcular_angulo(hombro, codo, muñeca)