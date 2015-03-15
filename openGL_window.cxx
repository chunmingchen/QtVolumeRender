/////////////////////////////////////////////////////
/// OpenGL example widget on Qt
/////////////////////////////////////////////////////

#include "cpp_headers.h"
#include "vtk_headers.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <openGL_window.h>

using namespace std;

openGL_window :: openGL_window(QWidget *parent)
 : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{

}

void openGL_window :: initializeGL()
{
    glClearColor(0, 0, 0, 1);
}

void openGL_window :: keyPressEvent(QKeyEvent* QE)
{
    if(QE->key() == Qt::Key_1)
    {
        cout<<"Coloring with I11"<<endl;
    }

    updateGL();
}

void openGL_window :: resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h); // set origin to bottom left corner
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //gluLookAt(0,0,5,0,0,0,0,1,0);
}

void openGL_window :: mousePressEvent(QMouseEvent *event)
{
    cout<<"clicked "<<event->x()<<"  "<<event->y()<<endl;
}

void openGL_window :: mouseMoveEvent(QMouseEvent *event)
{
    cout<<"dragging "<<event->x()<<"  "<<event->y()<<endl;
}

void openGL_window :: paintGL()
{
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_NICEST);
    glEnable(GL_LINE_SMOOTH);

    //glutSolidTeapot(1);
    glBegin(GL_POINTS);
    glColor3f(1,1,1);
    glVertex2f(0,0);
    glVertex2f(1,0);
    glVertex2f(0,1);
    glEnd();
    glPointSize(10);

    cout << "test " << endl;
    //glutSwapBuffers();


}
