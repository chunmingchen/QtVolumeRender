/////////////////////////////////////////////////////
//Soumya Dutta
//OSU CSE 2014
/////////////////////////////////////////////////////

#ifndef OPENGL_VTK_HEADER_H
#define OPENGL_VTK_HEADER_H

#include <cpp_headers.h>
#include <vtk_headers.h>
#include <QtOpenGL/QGLWidget>
#include <QMouseEvent>

class openGL_VTK_window : public QGLWidget
{
    Q_OBJECT

public:

    openGL_VTK_window(QWidget *parent);

    void paintGL();
    void initializeGL();
    void resizeGL(int w,int h);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *);

protected:

    QPoint lastPos;
    vtkNew<vtkRenderer> ren;
    vtkNew<vtkRenderWindow> renWin;
    vtkNew<vtkTransform> transform;
    int xform_mode = 0;
    const int XFORM_NONE  = 0;
    const int XFORM_ROTATE  = 1;
    const int XFORM_SCALE = 2;

    virtual void opengl_draw();
};

#endif
