#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "openGL_window.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //renderer = new openGL_window();


}

MainWindow::~MainWindow()
{
    delete ui;
}
