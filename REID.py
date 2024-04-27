import numpy as np
from typing import Optional, Dict, List
from torchreid.utils import FeatureExtractor
import cv2
from scipy.spatial.distance import cosine
import copy
class Person:
    def __init__(self,id,score,box,feature=None):
        self.id:int=id
        self.score:float=score
        self.box:tuple= copy.deepcopy(box)
        self.feature:Optional[np.array] = feature
        self.is_activated:bool=True
    
    def update(self,score,box,feature):
        self.score = score
        self.box:tuple = copy.deepcopy(box)
        self.feature = copy.deepcopy(feature)

    def activate(self):
        self.is_activated=True
        

    def reactivate(self):
        self.is_activated=True

    def marklost(self):
        self.is_activated=False

    

extractor = FeatureExtractor(model_name="osnet_x1_0",
                             model_path="reid_weights\\osnet__x0_1_market1501.pth",
                             device="cpu"
                             )

def crop(frame, box):
  """
  Crops a region from a frame in RGB format based on bounding box coordinates.

  Args:
      frame: A NumPy array representing the image frame (in BGR format).
      box: A tuple containing the top-left and bottom-right coordinates of the bounding box (x_min, y_min, x_max, y_max).

  Returns:
      A NumPy array representing the cropped region in RGB format, or None if the box is invalid.
  """
  box = tuple(box)
  (x_min, y_min, x_max, y_max) = box

  # Check for invalid box coordinates
  if x_min >= x_max or y_min >= y_max:
    return None

  # Clamp coordinates to frame dimensions
  x_min = int(max(0, x_min))
  y_min = int(max(0, y_min))
  x_max = int(min(frame.shape[1], x_max))
  y_max = int(min(frame.shape[0], y_max))

  # Convert to RGB format
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Crop the frame in RGB
  cropped_frame_rgb = frame_rgb[y_min:y_max, x_min:x_max]

  return cropped_frame_rgb


class REID:
    def __init__(self,threshold,feature_extractor):
        self.threshold = threshold
        self.active_persons:List[Person] = []
        self.lost_persons:List[Person] = []
        self.removed_persons = []
        self.frame_id:int = 0
        self.feature_extractor = feature_extractor
        self._p_id:int = 0

    def update(self,yolo_results,frame):
        #changed recently for testing
        yolo_results = copy.deepcopy(yolo_results)
        self.frame_id+=1
        overall_found:List[Person] = []
        if yolo_results.shape[1] == 5:
            scores = yolo_results[:, 4]
            bboxes = yolo_results[:, :4]
        else:
            yolo_results = yolo_results.cpu().numpy()
            scores = yolo_results[:, 4] * yolo_results[:, 5]
            bboxes = yolo_results[:, :4]  # x1y1x2y2
        
        if list(bboxes.shape)[0]==0:
            for p in self.active_persons:
                p.marklost()
            return []
                # self.sort_intolists(self.active_persons)

                
        high_conf_indices = scores > self.threshold
        high_conf_scores = scores[high_conf_indices]
        low_conf_indices = np.logical_and(scores<self.threshold,scores>0.1)
        ##### IMP: changing for testing...
        for p in self.active_persons:
            if p.score<self.threshold:
                p.marklost()
        #####
        high_conf_boxes = bboxes[high_conf_indices]
        # print("HCB:",high_conf_boxes)
        high_conf_regions = [crop(frame=frame,box=box) for box in high_conf_boxes]
        high_conf_features = [self.feature_extractor(region) for region in high_conf_regions]
        # print("high conf features:",1-cosine(high_conf_features[0].flatten(),high_conf_features[1].flatten()))

        active_persons:List[Person] = [p for p in self.active_persons]

        if len(active_persons)==0:
            for box,score,feature in zip(high_conf_boxes,high_conf_scores,high_conf_features):
                id = self.nextid()
                box = tuple(box)
                p = Person(id=id,score=score,box=box,feature=feature)
                p.activate()
                self.active_persons.append(p)
            print("p_boxes:",p.box)
            return [Person(id=p.id,score=p.score,box=p.box,feature=p.feature) for p in self.active_persons]
    

        active_features = [p.feature for p in active_persons]
        active_matching = self.compare_features(high_conf_features,active_features)
        matched, u_detection, u_person = self.filter_matches(active_matching)
        # print(active_matching)
        for row in matched:
            idet = row[0]
            iper = row[1]
            n_box, n_score, n_feature = (high_conf_boxes[idet], high_conf_scores[idet], high_conf_features[idet])
            # print("nboxes:",n_box,"idet:",idet)
            active_persons[iper].update(score=n_score,box=tuple(n_box),feature=n_feature)
            active_persons[iper].activate()

        for i_loosing in u_person:
            active_persons[i_loosing].marklost()

        for i_undet in u_detection:
            n_id, n_box, n_score, n_feature = (self.nextid(), high_conf_boxes[i_undet], high_conf_scores[i_undet], high_conf_features[i_undet])
            p = Person(n_id,n_score,tuple(n_box),n_feature)
            p.activate()
            self.active_persons.append(p)
        
        # print("testing here:",self.active_persons[-1].box)
        return [Person(p.id,p.score,p.box,p.feature) for p in self.active_persons if p.is_activated]


        ### First association with active_persons---------------###
        # recent_active_p = [Person(p.id,p.score,p.box,p.feature) for p in self.active_persons if (p.is_activated)]
        # recent_active_f = [p.feature for p in recent_active_p]
        # active_matching_costs = self.compare_features(high_conf_features,recent_active_f)
        # if active_matching_costs is None:
        #     for p in self.active_persons:
        #         p.marklost()
        #     self.sort_intolists()
        #     return []
        

        # matched, u_detection, u_active = self.filter_matches(active_matching_costs)

        # for idet, iact in matched:
        #     id = current_active_p[iact].id
        #     score = high_conf_scores[idet]
        #     box = high_conf_boxes[idet]
        #     feature = high_conf_features[idet]
        #     current_active_p[iact].activate()
        #     rtrn_p = Person(id=id,score=score,box=box,feature=feature)
        #     rtrn_p.activate()
        #     overall_found.append(rtrn_p)
        
        # for index in u_active:
        #     if index<len(current_active_p):
        #         current_active_p[index].marklost()

        # for p_found in overall_found:
        #     for p_active in self.active_persons:
        #         if p_found.id == p_active.id:
        #             p_active.update(p_found.score,p_found.box,p_found.feature)
        #             p_active.activate()
        
        
        

        # current_lost_p = [p for p in self.lost_persons if ((not p.is_activated) and p.feature!=None)]
        # current_lost_features_p = [p.feature for p in current_lost_p]
        # u_detection_features = [high_conf_features[i] for i in u_detection]
        # lost_matching_costs = self.compare_features(u_detection_features,current_lost_features_p)
        # if lost_matching_costs is None:
        #     self.sort_intolists()
        #     return overall_found
        # matched_second, u_detection_second, u_lost_second = self.filter_matches(lost_matching_costs)

        # for idet, ilost in matched_second:
        #     id = current_lost_p[ilost].id
        #     score = high_conf_scores[idet]
        #     box = high_conf_boxes[idet]
        #     feature = high_conf_features[idet]
        #     current_lost_p[ilost].activate()
        #     rtrn_p = Person(id=id,score=score,box=box,feature=feature)
        #     rtrn_p.activate()
        #     overall_found.append(rtrn_p)


        # for i in u_lost_second:
        #     if i<len(current_lost_p):
        #         current_lost_p[i].marklost()

        # for p_found in overall_found:
        #     for p_lost in self.lost_persons:
        #         if p_found.id == p_lost.id:
        #             p_lost.update(p_found.score,p_found.box,p_found.feature)
        #             p_lost.activate()
            
        # for i in u_detection_second:
        #     id = self.nextid()
        #     box = high_conf_boxes[i]
        #     score = high_conf_scores[i]
        #     feature = high_conf_features[i]
        #     active_insert_p = Person(id=id, score=score, box=box, feature=feature)
        #     active_insert_p.activate()
        #     rtrn_p = Person(id=id, score=score, box=box, feature=feature)
        #     rtrn_p.activate()
        #     self.active_persons.append(active_insert_p)
        #     overall_found.append(rtrn_p)
        
        # self.sort_intolists()
        # return overall_found
    
    
    # def sort_intolists(self):
    #     for i in range(len(self.lost_persons)):
    #         p = self.lost_persons[i]
    #         if p.is_activated:
    #             self.move_track(p.id,self.lost_persons,self.active_persons)
    #             i-=1
        
    #     for i in range(len(self.active_persons)):
    #         p = self.active_persons[i]
    #         if not p.is_activated:
    #             self.move_track(p.id,self.lost_persons,self.active_persons)
    #             i-=1
    #     for p in self.lost_persons:
    #         if p.is_activated:
    #             self.move_track(p.id,self.lost_persons,self.active_persons)
        
    #     for p in self.active_persons:
    #         if p.is_activated:
    #             self.move_track(p.id,self.active_persons,self.lost_persons)

    # def compare_features(self,features1:List[np.array],features2:List[np.array]):
    #     matching_costs = []
    #     for f1 in features1:
    #         row = []
    #         for f2 in features2:
    #             f1 = f1.flatten()
    #             f2 = f2.flatten()
    #             similarity = 1-cosine(f1,f2)
    #             row.append(similarity)
    #         matching_costs.append(row)
    #     return np.array(matching_costs)



    def compare_features(self, features1: List[np.array], features2: List[np.array]):
        # 1. Preprocessing
        def preprocess_feature(feature_array):
            return feature_array.flatten()  # Or other necessary preprocessing

        # Preprocess all features
        features1_flattened = [preprocess_feature(f) for f in features1]
        features2_flattened = [preprocess_feature(f) for f in features2]

        # 2. Similarity Calculation (Example using cosine similarity)
        def cosine_similarity(a, b):
            return 1-cosine(a,b)

        similarity_matrix = np.zeros((len(features1_flattened), len(features2_flattened)))
        for i in range(len(features1_flattened)):
            for j in range(len(features2_flattened)):
                similarity_matrix[i, j] = cosine_similarity(features1_flattened[i], features2_flattened[j])

        # 3. Extract Highest Matches
        return similarity_matrix

    import numpy as np

    def filter_matches(self,similarity_matrix):
        threshold = 0.65

        # 1. Matchings with highest similarity per row
        matches = []
        for row_index in range(similarity_matrix.shape[0]):
            row_values = similarity_matrix[row_index]
            high_sim_indices = np.where(row_values > threshold)[0]  # Indices with similarity > threshold

            if high_sim_indices.size > 0: 
                best_match_index = high_sim_indices[np.argmax(row_values[high_sim_indices])]  # Index of highest
                matches.append((row_index, best_match_index))

        # 2. Rows without high similarity matches (same as before)
        rows_without_matches = np.where(~np.any(similarity_matrix > threshold, axis=1))[0]

        # 3. Columns without high similarity matches (same as before)
        cols_without_matches = np.where(~np.any(similarity_matrix > threshold, axis=0))[0]

        return matches, rows_without_matches, cols_without_matches

    # def filter_matches(self,matrix:np.array):
    #     """Filters feature matches based on a similarity threshold.

    #     Args:
    #         matrix: A 2D NumPy array where matrix[i][j] represents the similarity
    #                 between feature i (from feature_list1) and feature j (from feature_list2).

    #     Returns:
    #         A tuple of:
    #             - matches_matrix: A 2D NumPy array with two columns (row_index, column_index)
    #                             for matches exceeding the similarity of 0.65.
    #             - unmatched_rows: A list of unmatched indices from the rows of the input matrix.
    #             - unmatched_cols: A list of unmatched indices from the columns of the input matrix.
    #     """

    #     # 1. Find matches exceeding the threshold 
    #     match_ind = np.where(matrix > self.threshold) # Get indices of matching elements
    #     matches_matrix = np.vstack(match_ind).T  # Combine them into the format (row_index, column_index)

    #     # 2. Find unmatched row indices
    #     all_row_indices = np.arange(matrix.shape[0])
    #     unmatched_rows = list(set(all_row_indices) - set(matches_matrix[:, 0]))

    #     # 3. Find unmatched column indices
    #     all_col_indices = np.arange(matrix.shape[1])
    #     unmatched_cols = list(set(all_col_indices) - set(matches_matrix[:, 1]))
        
    #     return matches_matrix, unmatched_rows, unmatched_cols
                
    def nextid(self):
        self._p_id+=1
        return self._p_id
    
                





        

        

