import math
from utils.messages import Messages
import cv2

class Geometry:
    def __init__(self, shaped):
        self.keypoints = shaped

    def calculate_angle(self, points):
        """
        This function calculates the angle between 2 lines formed by 3 points
            :param points: array of 3 values: the first, the second and the connection indexes of points
            :return: the angle as an integer
        """

        if len(points) != 3:
            Messages().error("The array of angle points has not 3 values!")
            pass

        else:
            # ----------- preparing the coordinates of the points ------
            index_point1 = points[0]
            index_connection_point = points[1]
            index_point2 = points[2]

            (y1, x1) = self.keypoints[index_point1][:2]
            (yc, xc) = self.keypoints[index_connection_point][:2]
            (y2, x2) = self.keypoints[index_point2][:2]

            # -------- calculating the slopes for the 2 lines -------
            m1 = self.slope_2_points((xc, yc), (x1, y1))
            m2 = self.slope_2_points((xc, yc), (x2, y2))

            # ------- calculating the angle -------
            angle_radians = math.atan((m2 - m1) / (1 + (m1 * m2)))
            angle_radians = math.fabs(angle_radians)
            angle_degrees = math.degrees(angle_radians)
            angle_degrees = Algebra().float_to_x_decimals(angle_degrees, 2)
            rounded_angle = self.round_by_base(int(angle_degrees), 5)

            # ------ making some adjustments --------
            are_points_collinear = self.collinearity_condition(self.keypoints[index_point1],
                                                               self.keypoints[index_point2],
                                                               self.keypoints[index_connection_point])

            if 177 <= angle_degrees <= 3 or are_points_collinear is True:
                rounded_angle = 0

            return int(rounded_angle)

    def slope_2_points(self, point1, point2):
        """
        This function calculates the slope of 2 numbers, using the the (y2 - y2) / (x2 - x1) formula
            :param point1: the first point as a variable of 2 values (x, y)
            :param point2: the second point as a variable of 2 values (x, y)
            :return: the slope value of the 2 points
        """

        slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        return slope

    def collinearity_condition(self, point1, point2, point3):
        """
        This function verifies if 3 given points are collinear or not
            :param point1: the first point as a variable of 3 values (y, x, confidence)
            :param point2: the second point as a variable of 3 values (y, x, confidence)
            :param point3: the third point as a variable of 3 values (y, x, confidence)
            :return: a boolean which verifies if the points are collinear or not
        """

        # ------ preparing the points coordinates -----
        (y1, x1) = point1[:2]
        (y2, x2) = point2[:2]
        (y3, x3) = point3[:2]

        # ------ calculating the 2 slopes that make the angle --------
        m = self.slope_2_points((x1, y1), (x2, y1))
        n = self.slope_2_points((x2, y2), (x3, y3))

        # ------- if 2 slopes have the same value, they are collinear => the points are collinear -----
        if m == n:
            return True

        return False

    def head_to_body_angle(self):
        (yr, xr) = self.keypoints[2][:2]  # the right eye keypoint coordinates
        (yl, xl) = self.keypoints[1][:2]  # the left eye keypoint coordinates

        mrl = self.slope_2_points((xr, yr), (xl, yl))
        mp = -1 / mrl

        (yr_shoulder, xr_shoulder) = self.keypoints[6][:2]
        (yl_shoulder, xl_shoulder) = self.keypoints[5][:2]
        m_shoulders = self.slope_2_points((xr_shoulder, yr_shoulder), (xl_shoulder, yl_shoulder))

        # ------- calculating the angle -------
        angle_radians = math.atan((mp - m_shoulders) / (1 + (mp * m_shoulders)))
        angle_radians = math.fabs(angle_radians)
        angle_degrees = math.degrees(angle_radians)
        angle_degrees = Algebra().float_to_x_decimals(angle_degrees, 2)
        rounded_angle = self.round_by_base(int(angle_degrees), 5)

        return rounded_angle

    def round_by_base(self, number, base=5):
        return base * round(number / base)

class Algebra:

    def float_to_x_decimals(self, number, x):
        """
        This functions helps in writing float numbers with a determined number of decimals after the comma
            :param number: the float number
            :param x: the number of decimals after the comma
            :return: a float with x decimals after the comma
        """

        number = int(number * math.pow(10, x))
        number = float(number / math.pow(10, x))

        return number
