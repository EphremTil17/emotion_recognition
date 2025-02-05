import cv2

def test_webcam():
    print("Testing webcam access...")
    print("Press 'q' to move to next camera, or 'ESC' to exit")
    
    # Try multiple camera indices
    for i in range(3):
        print(f"\nTesting camera index {i}")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"Camera {i} opened successfully")
            
            while True:
                ret, frame = cap.read()
                if ret:
                    # Add camera index label to frame
                    cv2.putText(frame, f"Camera {i}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Camera Test', frame)
                    
                    # Wait for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # Press 'q' to test next camera
                        break
                    elif key == 27:  # Press 'ESC' to exit
                        print("Exiting...")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                else:
                    print(f"Could not read frame from camera {i}")
                    break
        else:
            print(f"Could not open camera {i}")
            
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nTesting complete!")

if __name__ == "__main__":
    test_webcam()