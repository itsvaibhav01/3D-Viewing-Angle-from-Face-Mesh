import cv2
import math
import numpy as np
import matplotlib.pyplot as plt 
import mediapipe as mp

from custom.iris_lm_depth import from_landmarks_to_depth
from custom.face_geometry import PCF, get_metric_landmarks, procrustes_landmark_basis


class Proctor():    
    
    def __init__(self, frame_width, frame_height, channels):
        
        self.mp_face_mesh = mp.solutions.face_mesh
    
        # Prepare DrawingSpec for drawing the face landmarks later.
        self.mp_drawing = mp.solutions.drawing_utils 
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        self.points_idx = [33, 133, 362, 263, 61, 291, 199]
        self.points_idx = list(set(self.points_idx))
        self.points_idx.sort()

        self.left_eye_landmarks_id = np.array([33, 133])
        self.right_eye_landmarks_id = np.array([362, 263])

        self.dist_coeff = np.zeros((4, 1))

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.channels = channels
    
        self.image_size = (frame_width, frame_height)
        self.focal_length = frame_width

        # Head pose exlusive parameters
        center = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, center[0]], [0, self.focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        self.pcf = PCF(near=1, far=10000, frame_height=frame_height, frame_width=frame_width, fy=self.camera_matrix[1, 1])
    

    def get_stats(self, img, bool_results=False):
        """ 
        img: rgb image
        out: res_json
        """
        
        res_json = {
            'x-axis':-1,
            'left-eye-dist':-1,
            'right-eye-dist':-1,
            'left-eye-shape':-1,
            'right-eye-shape':-1  
        }
        
        RGB_img = img
        
        try:
            # Run MediaPipe Face Mesh.
            with self.mp_face_mesh.FaceMesh( static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5) as face_mesh:
                
                # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
                results = face_mesh.process(RGB_img)
                
                landmarks = None
                smooth_left_depth = -1
                smooth_right_depth = -1
                smooth_factor = 0.1

                if not results.multi_face_landmarks:
                    pass
                
                face_landmarks = results.multi_face_landmarks[0]
                
                frame_rgb = RGB_img
                results = face_mesh.process(frame_rgb)
            
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                landmarks = landmarks.T 
                
                ## head pose
                
                metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), self.pcf)
                model_points = metric_landmarks[0:3, self.points_idx].T
                
                image_points = (
                    landmarks[0:2, self.points_idx].T
                    * np.array([self.frame_width, self.frame_height])[None, :]
                )

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeff,
                    flags=cv2.cv2.SOLVEPNP_ITERATIVE,
                )
                
                ### Head pose string
                res_json['x-axis'] = round(rotation_vector[-1][0]*90, 2)
                
                ## head pose arrow 
                (nose_end_point2D, jacobian) = cv2.projectPoints(
                    np.array([(0.0, 0.0, 25.0)]),
                    rotation_vector,
                    translation_vector,
                    self.camera_matrix,
                    self.dist_coeff,
                )
                
                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                
                
                (left_depth, left_iris_size, left_iris_landmarks, left_eye_contours) = from_landmarks_to_depth(
                    frame_rgb,
                    landmarks[:, self.left_eye_landmarks_id],
                    self.image_size,
                    is_right_eye=False,
                    focal_length=self.focal_length,
                )

                (right_depth,right_iris_size,right_iris_landmarks,right_eye_contours) = from_landmarks_to_depth(
                    frame_rgb,
                    landmarks[:, self.right_eye_landmarks_id],
                    self.image_size,
                    is_right_eye=True,
                    focal_length=self.focal_length,
                )

                if smooth_right_depth < 0:
                    smooth_right_depth = right_depth
                else:
                    smooth_right_depth = (
                        smooth_right_depth * (1 - smooth_factor)
                        + right_depth * smooth_factor
                    )

                if smooth_left_depth < 0:
                    smooth_left_depth = left_depth
                else:
                    smooth_left_depth = (
                        smooth_left_depth * (1 - smooth_factor)
                        + left_depth * smooth_factor
                    )
                    
                ### Eyes stats 
                res_json['left-eye-dist'] = round(smooth_left_depth/10, 2)
                res_json['right-eye-dist'] = round(smooth_right_depth/10, 2)
                res_json['left-eye-shape'] = round(left_iris_size, 2)
                res_json['right-eye-shape'] = round(right_iris_size, 2)
                
        except Exception as e:
            print(e)
            results = None
            p1 = None
            p2 = None
            
                
        if bool_results:
            return res_json, results, p1, p2
        else:
            return res_json
                
        
    def get_view(self, img):
        
        stats, results, p1, p2 = self.get_stats(img, bool_results=True)
        
        eye_str = f"depth in cm: {stats['left-eye-dist']}, {stats['right-eye-dist']}"
        
        rt_txt = f"x-axis: {stats['x-axis']} degrees"  ## head pose x-angle

        annotated_image = img.copy()
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec)

        ## final prininting over image
        annotated_image = cv2.line(annotated_image, p1, p2, (255, 0, 0), 2)
        annotated_image = cv2.putText(annotated_image, text=rt_txt, org=(20,40),fontFace=2, fontScale=1, color=(170,50,60), thickness=1)
        annotated_image = cv2.putText(annotated_image, text=eye_str, org=(20,70),fontFace=2, fontScale=1, color=(170,50,100), thickness=1)
        
        # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return annotated_image
    

    
### Seperate function to run the script over webcam direclty
def procter_get_live_feed(mirror=True):
    
    cam = cv2.VideoCapture(0)

    ret_val, img = cam.read()
    
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frame_width, frame_height, channels = RGB_img.shape[1], RGB_img.shape[0], RGB_img.shape[2]

    watcher = Proctor(frame_width=frame_width, frame_height=frame_height, channels=channels)
    
    # running in all loop
    while True:
        ret_val, img = cam.read()
        
        if mirror: 
            img = cv2.flip(img, 1)

        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out_img = watcher.get_view(img=RGB_img)
        
        cv2.imshow('my webcam',cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    
if __name__ == '__main__':

    sample_index = 1
    img = cv2.imread(f'./custom/img/sample{sample_index}.jpeg')
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    watcher = Proctor(frame_width=640, frame_height=480, channels=3)
    obj = watcher.get_stats(img=RGB_img)

    #
    print('\nTEST\n', obj)
    print({'x-axis': -57.56, 'left-eye-dist': 91.3, 'right-eye-dist': 73.54, 'left-eye-shape': 8.31, 'right-eye-shape': 10.29})
    # 
