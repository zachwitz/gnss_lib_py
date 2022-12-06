import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image 
import pandas as pd

from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.utils.coordinates import ecef_to_el_az
from gnss_lib_py.utils.visualizations import plot_skyplot
from gnss_lib_py.parsers.android import AndroidDerived2021


path = "/Users/zachwitzel/Downloads/Photosphere - 2022Aug11 - For Derek and Zach/Sphere_Cross.jpg"
# path = "/Users/zachwitzel/Downloads/Photosphere_labels.png"
img = cv2.imread(path)
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.shape)
# img = img[:img.shape[0] // 2]
# print(img.shape)

def compute_elevation_azimuth(navdata, state_estimate):

    # check for receiver_state indexes
    rx_idxs = state_estimate.find_wildcard_indexes(["x_*_m","y_*_m",
                                                "z_*_m"],max_allow=1)

    if "el_sv_deg" not in navdata.rows or "az_sv_deg" not in navdata.rows:
        sv_el_az = None

    for timestamp, _, navdata_subset in navdata.loop_time("gps_millis"):

        pos_sv_m = navdata_subset[["x_sv_m","y_sv_m","z_sv_m"]].T

        # find time index for receiver_state NavData instance
        rx_t_idx = np.argmin(np.abs(state_estimate["gps_millis"] - timestamp))

        pos_rx_m = state_estimate[[rx_idxs["x_*_m"][0],
                                    rx_idxs["y_*_m"][0],
                                    rx_idxs["z_*_m"][0]],
                                    rx_t_idx].reshape(1,-1)

        timestep_el_az = ecef_to_el_az(pos_rx_m, pos_sv_m)

        if sv_el_az is None:
            sv_el_az = timestep_el_az.T
        else:
            sv_el_az = np.hstack((sv_el_az,timestep_el_az.T))

    navdata["el_sv_deg"] = sv_el_az[0,:]
    navdata["az_sv_deg"] = sv_el_az[1,:]



def rmat(alpha,
         beta,
         gamma):

    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha * np.pi / 180), -np.sin(alpha * np.pi / 180)],
            [0, np.sin(alpha * np.pi / 180), np.cos(alpha * np.pi / 180)],
        ]
    )
    ry = np.array(
        [
            [np.cos(beta * np.pi / 180), 0, np.sin(beta * np.pi / 180)],
            [0, 1, 0],
            [-np.sin(beta * np.pi / 180), 0, np.cos(beta * np.pi / 180)],
        ]
    )
    rz = np.array(
        [
            [np.cos(gamma * np.pi / 180), -np.sin(gamma * np.pi / 180), 0],
            [np.sin(gamma * np.pi / 180), np.cos(gamma * np.pi / 180), 0],
            [0, 0, 1],
        ]
    )

    return np.matmul(rz, np.matmul(ry, rx))

def equirect2Fisheye_UCM(
                          img,
                          outShape,
                          f=28, #focal length (28mm on Pixel 3)
                          xi=1.2, #distortion coefficient
                          angles=[0, 0, 0]
                          ):

    Hd = outShape[0]
    Wd = outShape[1]
    f = f
    xi = xi

    Hs, Ws = img.shape[:2]

    Cx = Wd / 2.0
    Cy = Hd / 2.0

    x = np.linspace(0, Wd - 1, num=Wd, dtype=np.float32)
    y = np.linspace(0, Hd - 1, num=Hd, dtype=np.float32)

    x, y = np.meshgrid(range(Wd), range(Hd))
    xref = 1
    yref = 1

    fmin = (
        np.lib.scimath.sqrt(
            -(1 - xi ** 2) *
              ((xref - Cx) ** 2 + (yref - Cy) ** 2)
        )
        * 1.0001
    )

    if xi ** 2 >= 1:
        fmin = np.real(fmin)
    else:
        fmin = np.imag(fmin)

# Originally no dividing F, this is a derived number, approximate
    x_hat = (x - Cx) / f / (-1.5 * xi) #-1.795
    y_hat = (y - Cy) / f / (1.5 * xi) #1.795

    x2_y2_hat = x_hat ** 2 + y_hat ** 2

    omega = np.real(
        xi + np.lib.scimath.sqrt(1 + (1 - xi ** 2) * x2_y2_hat)
    ) / (x2_y2_hat + 1)

    Ps_x = omega * x_hat
    Ps_y = omega * y_hat
    Ps_z = (omega - xi)
    
    # Roll, pitch, yaw
    alpha = angles[0]
    beta = angles[1]
    gamma = angles[2]

    R = np.matmul(
        rmat(alpha, beta, gamma),
        np.matmul(rmat(0, -90, 45), rmat(0, 90, 90)),
    )

    Ps = np.stack((Ps_x, Ps_y, Ps_z), -1)
    Ps = np.matmul(Ps, R.T)

    Ps_x, Ps_y, Ps_z = np.split(Ps, 3, axis=-1)
    Ps_x = Ps_x[:, :, 0]
    Ps_y = Ps_y[:, :, 0]
    Ps_z = Ps_z[:, :, 0]

    theta = np.arctan2(Ps_y, Ps_x)
    phi = np.arctan2(Ps_z, np.sqrt(Ps_x ** 2 + Ps_y ** 2))

    a = 2 * np.pi / (Ws - 1)
    b = np.pi - a * (Ws - 1)
    map_x = (1.0 / a) * (theta - b) #print this
    a = -np.pi / (Hs - 1)
    b = np.pi / 2
    # Originally this was (1.0 / a), tried changed to (0.5 / a), kind of worked
    map_y = (1 / a) * (phi - b) #print this
    print(np.max(map_x), np.max(map_y))
    #cv2.remap takes in map_x and map_y 
    output = cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )

    if f < fmin:
        r = np.sqrt(np.abs(-(f ** 2) / (1 - xi ** 2)))
        mask = np.zeros_like(output[:, :, 0])
        mask = cv2.circle(
            mask, (int(Cx), int(Cy)), int(r), (255, 255, 255), -1
        )
        output = cv2.bitwise_and(output, output, mask=mask)


    #added code to crop to size of the circle
    gray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = output[y:y+h,x:x+w]

    return crop

fish = equirect2Fisheye_UCM(img, (1000,1000), f=200, angles=[90, 45, 0])

fig, axs = plt.subplots()
plt.xticks(color='w')
plt.yticks(color='w')

fig.subplots_adjust(bottom=0.4)

im = axs.imshow(fish)


roll_slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
pitch_slider_ax = fig.add_axes([0.20, 0.05, 0.60, 0.03])
yaw_slider_ax = fig.add_axes([0.20, 0, 0.60, 0.03])
f_slider_ax = fig.add_axes([0.20, 0.15, 0.60, 0.03])
xi_slider_ax = fig.add_axes([0.20, 0.2, 0.60, 0.03])
save_ax = fig.add_axes([0.20, 0.25, 0.60, 0.03])


roll_slider = Slider(roll_slider_ax, "Roll", valinit=90, valmin=0, valmax=360)
pitch_slider = Slider(pitch_slider_ax, "Pitch", valinit=45, valmin=0, valmax=360)
yaw_slider = Slider(yaw_slider_ax, "Yaw", valmin=0, valinit=0, valmax=360)
xi_slider = Slider(xi_slider_ax, "X", valmin=0, valinit=1.2, valmax=3)
f_slider = Slider(f_slider_ax, "f", valmin=0, valinit=200, valmax=500)





def update(val):
    # axs.arrow(500,0,0,f_slider.val,width=1,color='red',length_includes_head=True)
    fish = equirect2Fisheye_UCM(img, (1000,1000), f=f_slider.val, angles=[roll_slider.val, pitch_slider.val, yaw_slider.val], xi=xi_slider.val)
    im = axs.imshow(fish)
    ax3.clear()
    ax3.patch.set_alpha(0)
    ax3.scatter(azim_arr / 360 * 2 * np.pi, elev_arr / 90 * (3/2 * f_slider.val))
    ax3.set_rmax(3 / 2 * f_slider.val)

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()


roll_slider.on_changed(update)
pitch_slider.on_changed(update)
yaw_slider.on_changed(update)
xi_slider.on_changed(update)
f_slider.on_changed(update)

def saving(event):
    new_img = Image.fromarray(equirect2Fisheye_UCM(img, (1000,1000), f=f_slider.val, angles=[roll_slider.val, pitch_slider.val, yaw_slider.val], xi=xi_slider.val))
    name = str(xi_slider.val) + "," + str(f_slider.val) + "," + str(roll_slider.val) + "," + str(pitch_slider.val) + "," + str(yaw_slider.val) +  ".png"
    new_img.save(name)

save_button = Button(save_ax, 'Save Fisheye Only', color='red')
save_button.on_clicked(saving)


navdata = AndroidDerived2021("/Users/zachwitzel/Downloads/Pixel4XL_derived.csv",
                            remove_timing_outliers=False)

state_estimate = solve_wls(navdata)
compute_elevation_azimuth(navdata, state_estimate)
elev_arr = np.array(navdata['el_sv_deg'])
azim_arr = np.array(navdata['az_sv_deg'])


ax3 = fig.add_subplot(projection='polar')
ax3.set_theta_zero_location("N")
ax3.patch.set_alpha(0)

ax3.scatter(azim_arr / 360 * 2 * np.pi, elev_arr / 90 * (3/2 * 200))
ax3.set_rmax(300)
rlabels = ax3.get_ymajorticklabels()
for label in rlabels:
    label.set_color('yellow')

plt.show()





fig, axs = plt.subplots()
plt.xticks(color='w')
plt.yticks(color='w')

# fig.subplots_adjust(bottom=0.4)


new_img = cv2.imread("/Users/zachwitzel/Downloads/Photosphere - 2022Aug11 - For Derek and Zach/SacramentoStreet-Tara/Fish-Eye/PXL_20220811_225004854.jpg")
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
plt.imshow(new_img)


ax3 = fig.add_subplot(projection='polar')
ax3.set_theta_zero_location("N")
ax3.patch.set_alpha(0)

ax3.scatter(azim_arr / 360 * 2 * np.pi, elev_arr / 90 * (3/2 * 200))
ax3.set_rmax(300)
rlabels = ax3.get_ymajorticklabels()
for label in rlabels:
    label.set_color('yellow')


plt.show()



# navdata = AndroidDerived2021("/Users/zachwitzel/Downloads/Pixel4XL_derived.csv",
#                             remove_timing_outliers=False)

# state_estimate = solve_wls(navdata)
# compute_elevation_azimuth(navdata, state_estimate)
# elev_arr = np.array(navdata['el_sv_deg'])
# azim_arr = np.array(navdata['az_sv_deg'])


# fig = plt.figure()
# ax = fig.add_subplot(projection='polar')

# ax.scatter(azim_arr / 360 * 2 * np.pi, elev_arr)
# ax.set_rmax(90)
# ax.set_rticks([30, 60, 90])  # Less radial ticks