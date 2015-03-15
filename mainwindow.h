#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "VRWidget.h"
#include "openGL_window.h"
#include "vtk_headers.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_btnLoad_clicked();

    void on_btnNext_clicked();

    void on_btnPrev_clicked();

private:
    Ui::MainWindow *ui;

    openGL_window* renderer;


    vtkSmartPointer<vtkImageData> data;
    vtkSmartPointer<vtkImageData> label;
    int file_idx = 1;
};

#endif // MAINWINDOW_H
