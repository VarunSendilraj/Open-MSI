import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# Global references to image objects
image_objects = []

def open_image_overlay():
    # Create the Tkinter window
    root = tk.Tk()
    root.title("Image Overlay")

    # Load the two images
    img1 = cv2.imread('GuiCode\\temp\mIMG.png')
    img2 = cv2.imread('GuiCode\\temp\genIMG.bmp')

    # Set opacity (0.0 - fully transparent, 1.0 - fully opaque)
    opacity = 0.5

    # Resize the images to desired dimensions
    desired_width = 600
    desired_height = 600
    img1 = cv2.resize(img1, (desired_width, desired_height))
    img2 = cv2.resize(img2, (desired_width, desired_height))

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGBA)
    img1[:, :, 3] = 255  # Set alpha channel to fully opaque
    img1 = Image.fromarray(img1)

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGBA)
    img2[:, :, 3] = int(opacity * 255)  # Set alpha channel opacity
    img2 = Image.fromarray(img2)

    # Create Tkinter-compatible images
    global img1_tk, img2_tk  # Declare as global variables
    img1_tk = ImageTk.PhotoImage(img1)
    img2_tk = ImageTk.PhotoImage(img2)

    # Create a canvas to display the images
    canvas_width = desired_width
    canvas_height = desired_height
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()

    # Display the images on the canvas
    canvas.img1_tk = img1_tk
    canvas.img2_tk = img2_tk

    # Display the images on the canvas
    image1_id = canvas.create_image(0, 0, anchor=tk.NW, image=img1_tk, tags="image1")
    image2_id = canvas.create_image(0, 0, anchor=tk.NW, image=img2_tk, tags="image2")
    image_objects.extend([img1_tk, img2_tk])  # Add image objects to the list

    def start_drag(event):
        canvas.old_x = event.x
        canvas.old_y = event.y

    # Mouse event function for dragging the movable image
    def move_image(event):
        canvas.move(image2_id, event.x - canvas.old_x, event.y - canvas.old_y)
        canvas.old_x = event.x
        canvas.old_y = event.y

    # Mouse event function for resizing the image
    def resize_image(event):
        global img2_tk_resized, canvas_width, canvas_height
        canvas_width = event.x
        canvas_height = event.y
        
        # Calculate the aspect ratio of the original image
        aspect_ratio = img2.width / img2.height
        
        # Calculate the new dimensions while maintaining the aspect ratio
        if canvas_width / canvas_height > aspect_ratio:
            canvas_width = int(canvas_height * aspect_ratio)
        else:
            canvas_height = int(canvas_width / aspect_ratio)
        
        resized_img2 = img2.resize((canvas_width, canvas_height), Image.ANTIALIAS)
        img2_tk_resized = ImageTk.PhotoImage(resized_img2)
        canvas.itemconfigure(image2_id, image=img2_tk_resized)

    canvas.bind("<ButtonPress-1>", start_drag)
    canvas.bind("<B1-Motion>", move_image)
    canvas.bind("<B3-Motion>", resize_image)

    # Rotate the image
    # Rotate the image
    def rotate_image(angle):
        global img2_tk_rotated, canvas_width, canvas_height
        rotated_img2 = img2.rotate(angle, resample=Image.BICUBIC)
        resized_rotated_img2 = rotated_img2.resize((canvas_width, canvas_height), Image.ANTIALIAS)
        img2_tk_rotated = ImageTk.PhotoImage(resized_rotated_img2)
        canvas.itemconfigure(image2_id, image=img2_tk_rotated)


    # Slider callback function
    def rotate_slider_callback(event):
        angle = rotate_slider.get()
        rotate_image(angle)

    # Create a slider for image rotation
    rotate_slider = tk.Scale(root, from_=0, to=360, orient=tk.HORIZONTAL, command=rotate_slider_callback)
    rotate_slider.pack(padx=10)

    # Save the overlaid image
    def save_image():
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")])
        if save_path:
            # Get the coordinates and dimensions of the canvas
            canvas_x = canvas.winfo_x()
            canvas_y = canvas.winfo_y()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            # Calculate the bounding box of the overlaid image
            overlaid_x = int(canvas.coords(image2_id)[0])
            overlaid_y = int(canvas.coords(image2_id)[1])
            overlaid_width = int(canvas_width)
            overlaid_height = int(canvas_height)

            # Calculate the position and dimensions of the resized and moved image
            resized_x = overlaid_x - canvas_x
            resized_y = overlaid_y - canvas_y
            resized_width = overlaid_width
            resized_height = overlaid_height

            # Create a new overlaid image with the updated dimensions
            overlaid_image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

            # Resize and paste the images onto the overlaid image with the updated coordinates
            resized_img1 = img1.resize((resized_width, resized_height), Image.ANTIALIAS)
            resized_img2 = img2.resize((resized_width, resized_height), Image.ANTIALIAS)

            # Create a composite image using the alpha channels of the resized images
            composite_image = Image.alpha_composite(resized_img1.convert("RGBA"), resized_img2.convert("RGBA"))

            overlaid_image.paste(composite_image, (resized_x, resized_y), mask=composite_image)

            overlaid_image.save(save_path)
            print("Overlayed image saved as", save_path)





    # Create a save button
    save_button = tk.Button(root, text="Save Overlayed Image", command=save_image)
    save_button.pack(padx=10)

    # Run the Tkinter event loop
    root.mainloop()

def overlayMain():
    open_image_overlay()

# Call overlayMain function with the path to the image
overlayMain()
