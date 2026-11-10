// main_window.h — main window with the Song Cover flow as the first tab.
#pragma once

#include <QMainWindow>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
};
