// main_window.cpp — main window with the Song Cover flow as the first tab.
#include "main_window.h"

#include "song_cover_page.h"

#include <QTabWidget>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("RVC-GGML UI"));

    auto* tabs = new QTabWidget(this);
    tabs->addTab(new SongCoverPage(tabs), QStringLiteral("Song Cover"));
    // "Multi-step" and "Training" tabs are planned; the scaffold keeps the
    // tab widget so they can be added without restructuring.
    setCentralWidget(tabs);
}
