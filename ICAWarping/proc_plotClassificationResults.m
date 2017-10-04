function proc_plotClassificationResults(x_labels, y_labels, user_number, classification_result)
% proc_plotClassificationResults - Plot classification results after
% data augmentation
%
%  This function generates line plots for classification results of each size of 
%  original training epochs after augmentation
%
%  Synopsis:
%  proc_plotClassificationResults(AUGMENTATION_SIZE, ORIGINAL_SIZE, USER_NUMBER, RESULTS)
%
%  Arguments:
%  AUGMENTATION_SIZE - Size of augmented epochs
%  ORIGINAL_SIZE - Sizes of original training epochs
%  USER_NUMBER - User's serial number
%  RESULTS - Classification results
%
%  Display: Line plots for different original epochs sizes

% Color codes for a maximum of 10 line plots
colors = [
    0         0    1.0000;
    0    1.0000    0.7586;
    1.0000    0.1034    0.7241;
    0         0    0.1724;
    0    1.0000         0;
    1.0000    0.8276         0;
    0    0.3448         0;
    0.5172    0.5172    1.0000;
    0.6207    0.3103    0.2759;
    1.0000         0         0];

figure;
data = classification_result;
color_counter = 1;
for data_size=100 * y_labels
    plot(x_labels, data(color_counter, :),  'color', colors(color_counter, :), 'LineWidth', 1, 'Marker', '.');
    legend_item{color_counter} = [num2str(data_size) '% training epochs' ];
    hold on;
    color_counter = color_counter + 1;
end
grid on;
hold off;
font = {'FontSize', 12, 'FontName', 'SansSerif'};
legendob = legend(legend_item);
set(legendob, 'FontSize', 12, 'FontName', 'SansSerif');
titleob = title(['Classification perf. vs augmented epochs generated using ICA warping of training epochs for user ' num2str(user_number)]);
set(titleob, 'FontSize', 12, 'FontName', 'SansSerif');
xob = xlabel('Number of augmented epochs');
set(xob, 'FontSize', 12, 'FontName', 'SansSerif');
yob = ylabel('Classification performance (in percentage)');
set(yob, 'FontSize', 12, 'FontName', 'SansSerif');

end
