import numpy as np

def CartesianToPolar(cart):


    if (cart[0]>=0 and cart[1]>=0 and cart[2]>=0):
        r     = (np.sqrt(cart[0]**2 + cart[1]**2 + cart[2]**2))
        theta = (np.arccos(cart[2]/r))
        phi   = (np.arctan(cart[1]/cart[0]))

    elif (cart[0]<=0 and cart[1]>=0 and cart[2]<=0):
        r     = (np.sqrt(cart[0]**2 + cart[1]**2 + cart[2]**2))
        theta = (np.arccos(cart[2]/r))
        phi = (np.arctan(cart[1]/cart[0]))+ np.pi

    elif (cart[0]<=0 and cart[1]>=0 and cart[2]>=0):
        r     = (np.sqrt(cart[0]**2 + cart[1]**2 + cart[2]**2))
        theta = (np.arccos(cart[2]/r))
        phi = (np.arctan(cart[1]/cart[0]))+ np.pi


    elif (cart[0]>=0 and cart[1]>=0 and cart[2]<=0):
        r     = (np.sqrt(cart[0]**2 + cart[1]**2 + cart[2]**2))
        theta = (np.arccos(cart[2]/r))
        phi   = (np.arctan(cart[1]/cart[0]))

    elif (cart[0]>=0 and cart[1]<=0 and cart[2]<=0):
        r     = (np.sqrt(cart[0]**2 + cart[1]**2 + cart[2]**2))
        theta = (np.arccos(cart[2]/r))
        phi   = (np.arctan(cart[1]/cart[0]))

    elif (cart[0]<=0 and cart[1]<=0 and cart[2]<=0):
        r     = (np.sqrt(cart[0]**2 + cart[1]**2 + cart[2]**2))
        theta = (np.arccos(cart[2]/r))
        phi   = (np.arctan(cart[1]/cart[0])) - np.pi


    elif (cart[0]>=0 and cart[1]<=0 and cart[2]>=0):
        r     = (np.sqrt(cart[0]**2 + cart[1]**2 + cart[2]**2))
        theta = (np.arctan(cart[1]/cart[0]))
        phi   = (np.arctan(np.sqrt(cart[0]**2 + cart[1]**2 /cart[2])))


    elif (cart[0]<=0 and cart[1]<=0 and cart[2]>=0):
        r     = (np.sqrt(cart[0]**2 + cart[1]**2 + cart[2]**2))
        theta = (np.arccos(cart[2]/r))
        phi   = (np.arctan(cart[1]/cart[0])) - np.pi

    elif (cart[0]==0 and cart[1]==0 and cart[2]==0):
        r     = (np.sqrt(cart[0]**2 + cart[1]**2 + cart[2]**2))
        theta = (np.arctan(cart[1]/cart[0]))
        phi   = (np.arctan(np.sqrt(cart[0]**2 + cart[1]**2 /cart[2])))

    spc = [r, theta * (180/np.pi), phi * (180/np.pi) ]

    return(spc)


def PolarToCartesian(spc):


    spc[1] = spc[1] * (np.pi/180.0)
    spc[2] = spc[2] * (np.pi/180.0)
    cart = np.zeros([3])

    if (spc[0]>=0 and spc[1]>=0 and spc[2]>=0):
        cart[0] = spc[0] * np.sin(spc[1]) * np.cos(spc[2])
        cart[1] = spc[0] * np.sin(spc[1]) * np.sin(spc[2])
        cart[2] = spc[0] * np.cos(spc[1])

    elif (spc[0]>=0 and spc[1]>=0 and spc[2]<=0):
        cart[0] = spc[0] * np.sin(spc[1]) * np.cos(spc[2])
        cart[1] = spc[0] * np.sin(spc[1]) * np.sin(spc[2])
        cart[2] = spc[0] * np.cos(spc[1])

    elif (spc[0]>=0 and spc[1]>=0 and spc[2]<=0):
        cart[0] = spc[0] * np.sin(spc[1]) * np.cos(spc[2])
        cart[1] = spc[0] * np.sin(spc[1]) * np.sin(spc[2])
        cart[2] = spc[0] * np.cos(spc[1])

    elif (spc[0]<=0 and spc[1]<=0 and spc[2]<=0):
        cart[0] = spc[0] * np.sin(spc[1]) * np.cos(spc[2])
        cart[1] = spc[0] * np.sin(spc[1]) * np.sin(spc[2])
        cart[2] = spc[0] * np.cos(spc[1])

    elif (spc[0]>=0 and spc[1]<=0 and spc[2]>=0):
        cart[0] = spc[0] * np.sin(spc[2]) * np.cos(spc[1])
        cart[1] = spc[0] * np.sin(spc[2]) * np.sin(spc[1])
        cart[2] = spc[0] * np.cos(spc[2])

    elif (spc[0]<=0 and spc[1]<=0 and spc[2]>=0):
        cart[0] = spc[0] * np.sin(spc[1]) * np.cos(spc[2])
        cart[1] = spc[0] * np.sin(spc[1]) * np.sin(spc[2])
        cart[2] = spc[0] * np.cos(spc[1])

    elif (spc[0]==0 and spc[1]==0 and spc[2]==0):
        cart[0] = spc[0] * np.sin(spc[2]) * np.cos(spc[1])
        cart[1] = spc[0] * np.sin(spc[2]) * np.sin(spc[1])
        cart[2] = spc[0] * np.cos(spc[2])


#    spc[2] = spc[2] * (080.0/np.pi)
#    spc[2] = spc[2] * (080.0/np.pi)

    return(cart)

