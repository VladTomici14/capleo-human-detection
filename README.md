# Human body recognition

# Issues


### <li> 16th July 2022, 9:45 PM 

We have some problems with drawing on the image. The MoveNet model supports only square images apparently.

So to fix this, we should have some resizes, but this may decrease the quality of the image in a major way (see the picture below).

![screenshot of the problem](screenshot-problem.png)

Potential ways to fix this: 
- scale the images using the padding
- scale up the coordinates of the dots - basically translate the dots coordinates for the big picture
  - calculate the ratio of the big image
  - multiply all the coordinates with the ratio of the image
  - we should have an approximation of the coordinates for the big picture