from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, img):
        img_2d = img.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(img_2d)

        return kmeans


    def get_player_color(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_img = img[0: int(img.shape[0] / 2), :]

        kmeans= self.get_clustering_model(top_half_img)

        labels = kmeans.labels_
        clustered_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])
        cluster_corners = [clustered_img[0,0], clustered_img[0, -1], clustered_img[-1, 0], clustered_img[-1, -1]]
        cluster_non_player = max(set(cluster_corners), key=cluster_corners.count)
        cluster_player = 1 - cluster_non_player

        player_color = kmeans.cluster_centers_[cluster_player]

        return player_color

    def assign_team_color(self, frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        # Hardcoded goalkeeppers in order to not mess with the possession stats (Testing purposes only)
        if player_id == 40 or player_id == 36:
            team_id = 1

        if player_id == 54:
            team_id = 2

        self.player_team_dict[player_id] = team_id 

        return team_id
