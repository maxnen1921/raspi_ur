import configSim


class UR:

    def __init__(self):
        # reference_point_coordinate_system
        self.RPX = configSim.RPX
        self.RPY = configSim.RPY

    def transform_coordinates_from_pixel(self, cx, cy):
        '''
        Transforming the pixel coordinates to meter values
        :param cx: x coordinate of detected DS
        :param cy: y coordinate of detected DS
        :return: returns x, y, z_rot coordinats for the robot
        '''
        # 22cm sind 188px in prod
        # 24.5cm sind 230px in sim
        x_coor = abs(self.RPX - cx) * 0.245 / 150
        y_coor = abs(self.RPY - cy) * 0.245 / 150

        x_coor = round(x_coor, 2)
        y_coor = round(y_coor, 2)

        z_rot = 0.0

        return x_coor, y_coor, z_rot
