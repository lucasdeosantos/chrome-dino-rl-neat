COLLISION_OFFSET_X = 1
COLLISION_OFFSET_Y = 1


class Rect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.right = x + width
        self.bottom = y + height

    def intersects(self, other):
        if self.right <= other.x or self.x >= other.right:
            return False
        
        if self.bottom <= other.y or self.y >= other.bottom:
            return False
        
        return True

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
        }


def position_hitboxes(layout, x, y):
    boxes = []
    for dx, dy, width, height in layout:
        boxes.append(Rect(x + dx, y + dy, width, height))
    
    return boxes
