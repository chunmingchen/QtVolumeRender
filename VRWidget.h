/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

#ifndef VRWIDGE_HEADER_H
#define VRWIDGE_HEADER_H

#include <cpp_headers.h>
#include "openGL_VTK_window.h"

class VRWidget : public openGL_VTK_window
{
    Q_OBJECT

public:

    VRWidget(QWidget *parent);

    //void paintGL();
    void initializeGL();
    void resizeGL(int w,int h);
    //void mousePressEvent(QMouseEvent *event);
    //void mouseMoveEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *);

protected:
    virtual void opengl_draw();

};

#endif
