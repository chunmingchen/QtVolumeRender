/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

#include "cpp_headers.h"
#include "vtk_headers.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include "VRWidget.h"

using namespace std;

VRWidget :: VRWidget(QWidget *parent)
 : openGL_VTK_window(parent)
{

}

void VRWidget :: initializeGL()
{
    openGL_VTK_window::initializeGL();
}

void VRWidget :: keyPressEvent(QKeyEvent* QE)
{
    if(QE->key() == Qt::Key_1)
    {
        cout<<"Coloring with I11"<<endl;
    }

    updateGL();
}

void VRWidget :: resizeGL(int w, int h)
{
   openGL_VTK_window::resizeGL(w, h) ;
}

void VRWidget:: opengl_draw()
{

}
