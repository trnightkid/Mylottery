
-- MySQL数据库表结构
CREATE TABLE lottery_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    issue_number INT NOT NULL COMMENT '期号',
    red_ball_1 TINYINT UNSIGNED NOT NULL COMMENT '红球1',
    red_ball_2 TINYINT UNSIGNED NOT NULL COMMENT '红球2',
    red_ball_3 TINYINT UNSIGNED NOT NULL COMMENT '红球3',
    red_ball_4 TINYINT UNSIGNED NOT NULL COMMENT '红球4',
    red_ball_5 TINYINT UNSIGNED NOT NULL COMMENT '红球5',
    red_ball_6 TINYINT UNSIGNED NOT NULL COMMENT '红球6',
    blue_ball TINYINT UNSIGNED NOT NULL COMMENT '蓝球',
    jackpot BIGINT UNSIGNED NOT NULL COMMENT '奖池奖金(元)',
    first_prize_count INT UNSIGNED NOT NULL COMMENT '一等奖注数',
    first_prize_amount BIGINT UNSIGNED NOT NULL COMMENT '一等奖奖金(元)',
    second_prize_count INT UNSIGNED NOT NULL COMMENT '二等奖注数',
    second_prize_amount INT UNSIGNED NOT NULL COMMENT '二等奖奖金(元)',
    total_stake BIGINT UNSIGNED NOT NULL COMMENT '总投注额(元)',
    draw_date DATE NOT NULL COMMENT '开奖日期',
    INDEX idx_issue_number (issue_number),
    INDEX idx_draw_date (draw_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='彩票开奖数据';

-- 导入数据命令示例:
-- LOAD DATA INFILE 'lottery_data.csv' 
-- INTO TABLE lottery_data 
-- FIELDS TERMINATED BY ',' 
-- ENCLOSED BY '"' 
-- LINES TERMINATED BY '\n' 
-- IGNORE 1 ROWS;
