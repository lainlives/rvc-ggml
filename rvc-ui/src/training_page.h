// training_page.h — RVC training: preprocess -> train -> export.
//
// Three independently runnable cards share the experiment paths at the top:
//   rvc-prep    audio folder -> training cache (--exp)
//   rvc-train   cache + pretrained/fresh G,D -> checkpoints, snapshots, periodic exports
//   rvc-train --export-gguf   training-state checkpoint -> inference GGUF
//
// The two backends (ggml for prep, traincpp for training) must never coexist in
// one process (project rule R28), so every step is a separate subprocess.
#pragma once

#include <QString>
#include <QVector>
#include <QWidget>

#include <functional>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QProcess;
class QPushButton;
class QSpinBox;

class TrainingPage : public QWidget {
    Q_OBJECT
public:
    explicit TrainingPage(QWidget* parent = nullptr);
    ~TrainingPage() override;

private:
    void build_ui();
    void populate_models();
    void log(const QString& text);
    void run_process(const QString& program, const QStringList& args, QLabel* status,
                     QPushButton* button, const QString& start_text,
                     std::function<void(bool)> done = {});
    QString cache_dir() const;
    QString out_dir() const;
    QString config_path() const;
    QString find_training_state(const QString& version, QChar which) const;
    void sync_config();
    void sync_pretrain_paths();
    void refresh_export_sources();
    void update_export_path();

    void run_prep();
    void run_train();
    void run_export();

    // Experiment paths (shared by the cards).
    QLineEdit* cache_edit_ = nullptr;
    QLineEdit* out_edit_ = nullptr;
    QComboBox* config_combo_ = nullptr;

    // Preprocess (rvc-prep).
    QLineEdit* input_edit_ = nullptr;
    QComboBox* version_combo_ = nullptr;
    QComboBox* prep_extractor_combo_ = nullptr;
    QComboBox* prep_f0_combo_ = nullptr;
    QComboBox* prep_device_combo_ = nullptr;
    QPushButton* prep_run_ = nullptr;
    QLabel* prep_status_ = nullptr;

    // Train (rvc-train --train).
    QComboBox* mode_combo_ = nullptr;
    QLineEdit* g_edit_ = nullptr;
    QLineEdit* d_edit_ = nullptr;
    QLineEdit* resume_edit_ = nullptr;
    QSpinBox* epochs_spin_ = nullptr;
    QSpinBox* batch_spin_ = nullptr;
    QSpinBox* bucket_spin_ = nullptr;
    QSpinBox* disc_len_spin_ = nullptr;
    QSpinBox* save_every_spin_ = nullptr;
    QSpinBox* export_every_spin_ = nullptr;
    QSpinBox* grad_accum_spin_ = nullptr;
    QComboBox* train_device_combo_ = nullptr;
    QPushButton* train_run_ = nullptr;
    QLabel* train_status_ = nullptr;

    // Export (rvc-train --export-gguf).
    QComboBox* ckpt_combo_ = nullptr;
    QLineEdit* export_edit_ = nullptr;
    QPushButton* export_run_ = nullptr;
    QLabel* export_status_ = nullptr;

    QPlainTextEdit* log_view_ = nullptr;

    bool g_edited_ = false;
    bool d_edited_ = false;
    bool export_edited_ = false;
    QVector<QProcess*> active_;
};
