import cv2

# Load input image
img = cv2.imread("img/pexelsylanitekoppens23431701.jpg")

# Check if the image is loaded properly
if img is None:
    print("Error: Could not load the input image.")
else:
    (H, W) = img.shape[:2]

    # Prepare input for the network
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H),
                                 swapRB=False, crop=False)

    # Load model files
    net = cv2.dnn.readNetFromCaffe(r"C:\\file\\deploy.prototxt",
                                    r"C:\\file\\hed_pretrained_bsds.caffemodel")

    # Pass the blob through the network
    net.setInput(blob)
    hed = net.forward()

    # Resize output to match input size
    hed = cv2.resize(hed[0, 0], (W, H))

    # Convert to 0-255 range
    hed = (255 * hed).astype("uint8")

    # Display the result
    cv2.imshow("Input", img)
    cv2.imshow("HED", hed)

    # ✅ Print message in console
    print("Holistically Nested Edge Detected")

    # ✅ Close windows automatically after 5 seconds
    cv2.waitKey(5000)  # 5000 ms = 5 seconds
    cv2.destroyAllWindows()

    # ✅ Close input file properly
    print("Closing input file and output window.")