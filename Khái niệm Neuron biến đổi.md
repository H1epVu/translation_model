CHƯƠNG I. MẠNG NƠ-RON
1.1. Khái niệm
-	Mạng neural được xây dựng dựa trên mạng neural sinh học. Nó gồm các neural (nút) nối với nhau và xử lý thông tin bằng cách truyền theo các kết nối và tính giá trị tại các nút.
1.2. Kiến trúc
-	Mạng neuron với mỗi nút sẽ có những dữ liệu đầu vào, biến đổi những dữ liệu đầu vào này bằng cách tính tổng các input với weight tương ứng trên các đầu vào, sau đó áp dụng một hàm biến đổi phi tuyến tính cho phép biến đổi này để tính toán trạng thái trung gian. 3 bước trên tạo thành 1 lớp và hàm biến đổi còn được gọi là activation funtion. Các output của layer này sẽ là input của layer phía sau.
-	Thông qua việc lặp lại các bước trên, neural-network học thông qua nhiều layer và các nút phi tuyến tính rồi sau đó kết hợp lại ở layer cuối cùng để cho ra 1 dự đoán.
-	Neural-network học bằng cách tạo ra các tín hiệu lỗi đo lường sự khác biệt giữa các dự đoán của mạng và giá trị mong muốn, sau đó sử dụng tín hiệu lỗi này để cập nhật lại weight và bias trong activation function để việc dự đoán sau đó chính xác hơn.
 <img width="355" height="321" alt="image" src="https://github.com/user-attachments/assets/ac482abc-9caa-4968-88be-20daec82df0d" />
Hình 1.1. Mạng neuron 1 lớp ẩn
 <img width="686" height="344" alt="image" src="https://github.com/user-attachments/assets/6435a9c4-2a0f-47c3-9b72-56e1f656aef8" />

1.3. Thành phần
1.3.1. Activation function
-	Activation function là 1 thành phần rất quan trọng của neural-network. Nó quyết định khi nào thì 1 neuron được kích hoạt hoặc không. Liệu thông tin mà neuron nhận được có liên quan đến thông tin được đưa ra hay nên bỏ qua.
 <img width="595" height="103" alt="image" src="https://github.com/user-attachments/assets/6271a393-c563-4701-9ea8-37d5e345a225" />
-	Activation function là 1 phép biến đổi phi tuyến tính mà chúng ta thực hiện đối với tín hiệu đầu vào. Đầu ra được chuyển đổi này sẽ được sử dụng làm đầu vào của neuron ở layer tiếp theo.
-	Nếu không có activation function thì weight và bias chỉ đơn giản như 1 hàm biến đổi tuyến tính. Giải 1 hàm tuyến tính sẽ đơn giản hơn nhiều nhưng sẽ khó có thể mô hình hóa và giải được những vấn đề phức tạp. Một mạng neuron nếu không có activation function thì cơ bản chỉ là 1 model hồi quy tuyến tính. Activation function thực hiện việc biến đổi phi tuyến tính với đầu vào làm việc học hỏi và thực hiện những nhiệm vụ phức tạp hơn như dịch ngôn ngữ hoặc phân loại ảnh là khả thi.
-	Activation function hỗ trợ back-propagation (tuyên truyền ngược) với việc cung cấp các lỗi để có thể cập nhật lại các weight và bias, việc này giúp mô hình có khả năng tự hoàn thiện.
1.3.1.1. Một số hàm activation phổ biến
 <img width="491" height="409" alt="image" src="https://github.com/user-attachments/assets/b4a5b45b-86df-4144-8eee-3761f9cd0a75" />
Hính 1.3. Các hàm active
1.3.1.2. Lựa chọn
-	Các hàm sigmoid và sự kết hợp của chúng thường phù hợp với những bài toán phân loại
-	Sigmoid và tanh đôi khi nên tránh sử dụng đồng thời vì có thể khiến gradient biến mất
-	ReLU là 1 activation function phổ biến và thường dùng nhất hiện nay o Nếu gặp những trường hợp có tế bào neuron chết trong mạng thì leaky thì ReLU là 1 lựa chọn hoàn hảo
-	ReLU function chỉ có thể được sử dụng trong những hidden layer
 
1.3.2. Convolution
1.3.2.1. Khái niệm
-	Bộ lọc với điểm ảnh để trích xuất các đặc tính từ ảnh đầu vào, duy trì mối liên kết giữa các pixel bằng cách tìm hiểu đặc tính của ảnh và sử dụng các ô nhỏ của dữ liệu đầu vào.
-	Convolution Layer : là một loạt feature map được trích xuất từ ảnh ban đầu.
-	Convolution Filter (kernel): sẽ có nhiều bộ lọc khác nhau như là: Phát hiện cạnh của ảnh, làm mờ, làm sắc nét,… chúng ta có thể áp dụng các bộ lọc trên trong các trường hợp cụ thể mà mình mong muốn.
1.3.2.2. Các bước thực hiện
-	Chúng ta sẽ chuyển ảnh ban đầu về ma trận có giá trị 0,1.
-	Từ ma trận ảnh ban đầu đã có và ma trận bộ lọc (kernel) chúng ta tích chập hai ma trận thành một ma trận đặc điểm của ảnh (feature map).
-	Ma trận đầu vào có kích thước là H1 x W1 x D (H = height, W = width, D = dimension) và bộ lọc (kernel) là H2 x W2 x D thì ma trận đặc điểm ảnh sẽ là:
 <img width="397" height="160" alt="image" src="https://github.com/user-attachments/assets/01be6310-0db0-4b0a-9deb-927a808cb4eb" />
-	Ví dụ ma trận đặc điểm ảnh tổng quát
 <img width="438" height="313" alt="image" src="https://github.com/user-attachments/assets/4a52a793-0ba3-4ce8-b24d-d638841e52e9" />
Hình 1.5. Ma trận đặc điểm ảnh
-	Ví dụ cụ thể về tính ma trận đặc điểm ảnh
 <img width="712" height="219" alt="image" src="https://github.com/user-attachments/assets/a3e87dd4-af61-4eca-ab8e-3a661e93417c" />
Hình 1.6. Tính ma trận điểm ảnh
1.4. Phân loại
1.4.1. Mạng nơ-ron truyền thẳng
-	Mạng nơ-ron truyền thẳng xử lý dữ liệu theo một chiều, từ nút đầu vào đến nút đầu ra. Mỗi nút trong một lớp được kết nối với tất cả các nút trong lớp tiếp theo. Mạng truyền thẳng sử dụng một quy trình phản hồi để cải thiện dự đoán theo thời gian.
1.4.2. Thuật toán truyền ngược
-	Mạng nơ-ron nhân tạo liên tục học hỏi bằng cách sử dụng vòng lặp phản hồi hiệu chỉnh để cải thiện phân tích dự đoán của chúng. Đơn giản mà nói, bạn có thể coi rằng dữ liệu truyền từ nút đầu vào đến nút đầu ra qua nhiều lối đi khác nhau trong mạng nơ-ron. Chỉ có duy nhất một lối đi chính xác, ánh xạ nút đầu vào đến nút đầu ra thích hợp. Để tìm ra lối đi này, mạng nơ-ron sử dụng một vòng lặp phản hồi với cách thức hoạt động như sau:
•	Mỗi nút đưa ra một dự đoán về nút tiếp theo trên lối đi.
•	Nút này sẽ kiểm tra tính chính xác của dự đoán. Các nút sẽ chỉ định giá trị trọng số cao hơn cho những lối đi tới nhiều dự đoán chính xác hơn và giá trị trọng số thấp hơn cho các lối đi tới dự đoán không chính xác.
•	Đối với điểm dữ liệu tiếp theo, các nút đưa ra dự đoán mới bằng cách sử dụng các lối đi có trọng số cao hơn rồi lặp lại Bước 1.
1.4.3. Mạng nơ-ron tích chập
-	Những lớp ẩn trong mạng nơ-ron tích chập thực hiện các chức năng toán học cụ thể, như tóm tắt hoặc sàng lọc, được gọi là tích chập. Chúng rất hữu ích trong việc phân loại hình ảnh vì chúng có thể trích xuất các đặc điểm liên quan từ hình ảnh, điều này có lợi cho việc nhận dạng và phân loại hình ảnh. Biểu mẫu mới dễ xử lý hơn mà không làm mất đi các đặc điểm quan trọng để đưa ra dự đoán chính xác. Mỗi lớp ẩn trích xuất và xử lý các đặc điểm hình ảnh khác nhau, như các cạnh, màu sắc và độ sâu.

 
CHƯƠNG 2. MẠNG NEURON HỒI QUY (RNN)
2.1. Khái niệm
-	Mạng nơ-ron hồi quy (RNN) là một mô hình học sâu được đào tạo để xử lý và chuyển đổi đầu vào dữ liệu tuần tự thành đầu ra dữ liệu tuần tự cụ thể.
-	Dữ liệu tuần tự là dữ liệu, chẳng hạn như từ, câu hoặc dữ liệu chuỗi thời gian, trong đó các thành phần tuần tự tương quan với nhau dựa trên ngữ nghĩa phức tạp và quy tắc cú pháp.
-	RNN là một hệ thống phần mềm gồm nhiều thành phần được kết nối với nhau theo cách con người thực hiện chuyển đổi dữ liệu tuần tự, chẳng hạn như dịch văn bản từ ngôn ngữ này sang ngôn ngữ khác. Phần lớn RNN đang được thay thế bằng trí tuệ nhân tạo (AI) dựa trên công cụ biến đổi và các mô hình ngôn ngữ lớn (LLM), hiệu quả hơn nhiều trong việc xử lý dữ liệu tuần tự.

2.2. Cách hoạt động
 <img width="686" height="352" alt="image" src="https://github.com/user-attachments/assets/82a28a85-2e51-4126-99b6-cb3fd2ffe1bc" />
Hình 2.1. Sơ đồ hoạt động
RNN được tạo thành từ các nơ-ron: các nút xử lý dữ liệu kết hợp cùng nhau để thực hiện các tác vụ phức tạp. Các nơ-ron được tổ chức dưới dạng lớp đầu vào, đầu ra và ẩn. Lớp đầu vào nhận thông tin để xử lý và lớp đầu ra cung cấp kết quả. Quá trình xử lý dữ liệu, phân tích và dự đoán diễn ra trong lớp ẩn.
2.2.1. Lớp ẩn
-	RNN hoạt động bằng cách lần lượt truyền dữ liệu tuần tự nhận được đến các lớp ẩn. Tuy nhiên, RNN cũng có quy trình làm việc tự lặp lại hay hồi quy: lớp ẩn có thể ghi nhớ và sử dụng các đầu vào trước đó cho các dự đoán trong tương lai trong một thành phần bộ nhớ ngắn hạn. Quy trình này sử dụng đầu vào hiện tại và bộ nhớ đã lưu trữ để dự đoán chuỗi tiếp theo. 
-	Ví dụ: hãy xem xét chuỗi: Apple is red (Táo màu đỏ). Bạn muốn RNN dự đoán red (màu đỏ) khi nhận được chuỗi đầu vào Apple is (Táo màu). Khi xử lý từ Apple (Táo), lớp ẩn sẽ lưu trữ một bản sao trong bộ nhớ. Tiếp theo, khi thấy từ is (màu), lớp ẩn gọi lại Apple (Táo) từ bộ nhớ của mình và hiểu toàn bộ chuỗi: Apple is (Táo màu) là ngữ cảnh. Sau đó, lớp ẩn có thể dự đoán red (màu đỏ) để cải thiện độ chính xác. Do đó, RNN trở nên hữu ích trong nhận dạng giọng nói, dịch máy và các tác vụ lập mô hình ngôn ngữ khác.

2.2.2. Đào tạo
-	Các kỹ sư máy học (ML) đào tạo các mạng nơ-ron sâu như RNN bằng cách cung cấp dữ liệu đào tạo cho mô hình và tinh chỉnh hiệu năng của mô hình. Trong ML, trọng số của nơ-ron là tín hiệu để xác định mức độ ảnh hưởng của thông tin đã học trong quá trình đào tạo khi dự đoán đầu ra. Mỗi lớp trong RNN đều có trọng số bằng nhau. 
-	Các kỹ sư ML điều chỉnh trọng số để dự đoán chính xác hơn. Họ sử dụng một kỹ thuật gọi là truyền ngược qua thời gian (BPTT) để tính lỗi mô hình và điều chỉnh trọng số của mô hình cho phù hợp. BPTT khôi phục đầu ra về bước thời gian trước và tính lại tỷ lệ lỗi. Qua đó, kỹ thuật này có thể xác định trạng thái ẩn nào trong chuỗi đang gây ra lỗi đáng kể và điều chỉnh lại trọng số để giảm biên lỗi.

2.3. Phân loại
2.3.1. Một-nhiều
-	Loại RNN này dẫn một đầu vào đến một số đầu ra. Loại này tạo điều kiện cho các ứng dụng ngôn ngữ như chú thích hình ảnh bằng cách tạo một câu từ một từ khóa duy nhất.
2.3.2. Nhiều-nhiều
-	Mô hình sử dụng nhiều đầu vào để dự đoán nhiều đầu ra. Ví dụ: bạn có thể tạo một công cụ dịch ngôn ngữ bằng RNN, với khả năng phân tích câu và cấu trúc chính xác các từ trong một ngôn ngữ khác. 
2.3.3. Nhiều-một
-	Một số đầu vào được ánh xạ đến một đầu ra. Loại này rất hữu ích trong các ứng dụng như phân tích cảm xúc, trong đó mô hình dự đoán cảm xúc của khách hàng như tích cực, tiêu cực và trung lập từ lời chứng thực đầu vào.

2.4. So sánh với các mạng học sâu khác
2.4.1. Mạng nơ-ron hồi quy so với mạng nơ-ron truyền thẳng
-	Giống như RNN, mạng nơ-ron truyền thẳng là mạng nơ-ron nhân tạo truyền thông tin từ đầu này sang đầu kia của kiến trúc. Mạng nơ-ron truyền thẳng có thể thực hiện các nhiệm vụ phân loại, hồi quy hoặc nhận dạng đơn giản nhưng không thể nhớ đầu vào trước đó đã được mạng xử lý. Ví dụ: mạng này quên Apple (Táo) vào thời điểm nơ-ron của mạng xử lý từ is (là). RNN khắc phục được hạn chế bộ nhớ này bằng cách đưa trạng thái bộ nhớ ẩn vào nơ-ron.

2.4.2. Mạng nơ-ron hồi quy so với mạng nơ-ron tích chập
-	Mạng nơ-ron tích chập là mạng nơ-ron nhân tạo được thiết kế để xử lý dữ liệu chuỗi thời gian. Bạn có thể sử dụng mạng nơ-ron tích chập để trích xuất thông tin không gian từ video và hình ảnh bằng cách truyền thông tin đó qua một loạt các lớp tích chập và tổng hợp trong mạng nơ-ron. RNN được thiết kế để ghi lại các phần phụ thuộc lâu dài trong dữ liệu tuần tự

2.5. Hạn chế
Độ dốc cực lớn
-	RNN có thể dự đoán sai đầu ra trong khóa đào tạo ban đầu. Bạn cần lặp lại nhiều lần để điều chỉnh các thông số của mô hình nhằm giảm tỷ lệ lỗi. Bạn có thể mô tả độ nhạy của tỷ lệ lỗi tương ứng với thông số của mô hình dưới dạng độ dốc. Bạn có thể hình dung độ dốc như một đường dốc mà bạn đi xuống từ một ngọn đồi. Độ dốc lớn hơn cho phép mô hình học nhanh hơn và độ dốc nhỏ làm giảm tốc độ học tập.
-	Độ dốc cực lớn xuất hiện khi độ dốc tăng theo cấp số nhân cho đến khi RNN trở nên không ổn định. Khi độ dốc trở nên lớn vô hạn, RNN hoạt động thất thường, dẫn đến các vấn đề về hiệu năng như quá khớp. Quá khớp là hiện tượng mô hình có thể dự đoán chính xác với dữ liệu đào tạo nhưng không thể dự đoán chính xác với dữ liệu thực tế. 
Độ dốc biến mất
-	Bài toán độ dốc biến mất là một điều kiện, trong đó độ dốc của mô hình đạt đến 0 trong quá trình đào tạo. Khi độ dốc biến mất, RNN không học hiệu quả từ dữ liệu đào tạo, dẫn đến chưa khớp. Mô hình chưa khớp không thể hoạt động tốt trong các ứng dụng thực tế vì các trọng số chưa được điều chỉnh thích hợp. RNN sẽ có rủi ro gặp phải vấn đề độ dốc cực lớn và biến mất khi xử lý các chuỗi dữ liệu dài. 
Thời gian đào tạo chậm
-	RNN xử lý dữ liệu tuần tự, do đó hạn chế khả năng xử lý khối lượng lớn văn bản một cách hiệu quả. Ví dụ: mô hình RNN có thể phân tích cảm xúc của người mua từ một vài câu. Tuy nhiên, mô hình này yêu cầu phải có năng lực điện toán cực lớn, không gian bộ nhớ và thời gian để tóm tắt một trang của một bài luận.

2.6. Khắc phục những hạn chế
Bộ chuyển đổi là các mô hình học sâu sử dụng các cơ chế tự chú ý trong mạng nơ-ron truyền thẳng bộ mã hóa-giải mã. Bộ chuyển đổi có thể xử lý dữ liệu tuần tự theo cách giống như RNN. 
2.6.1. Tự chú ý
-	Bộ chuyển đổi không sử dụng trạng thái ẩn để ghi lại các phần phụ thuộc lẫn nhau của chuỗi dữ liệu. Thay vào đó, họ sử dụng đầu tự chú ý để xử lý song song các chuỗi dữ liệu. Điều này cho phép bộ chuyển đổi đào tạo và xử lý các chuỗi dài hơn trong thời gian ngắn hơn so với RNN. Với cơ chế tự chú ý, bộ chuyển đổi khắc phục được những hạn chế về bộ nhớ và các phần phụ thuộc lẫn nhau của chuỗi mà RNN gặp phải. Bộ chuyển đổi có thể xử lý song song các chuỗi dữ liệu và sử dụng mã hóa vị trí để ghi nhớ cách mỗi đầu vào liên hệ với các đầu vào khác. 
2.6.2. Tính song song
-	Bộ chuyển đổi giải quyết các vấn đề về độ dốc mà RNN gặp phải bằng cách cho phép tính song song trong quá trình đào tạo. Nhờ xử lý đồng thời tất cả các chuỗi đầu vào nên bộ chuyển đổi không gặp phải những hạn chế truyền ngược vì các độ dốc có thể tự do di chuyển đến tất cả các trọng số. Độ dốc cũng được tối ưu hóa cho điện toán song song, do các đơn vị xử lý đồ họa (GPU) cung cấp để phát triển AI tạo sinh. Tính song song cho phép bộ chuyển đổi điều chỉnh quy mô cực kỳ lớn và xử lý các tác vụ NLP phức tạp bằng cách xây dựng các mô hình lớn hơn.

 
CHƯƠNG 3. MẠNG NƠ RON BIẾN ĐỔI 
3.1. Khái niệm
-	Transformer là một kiến trúc mạng nơ-ron nhân tạo (AI) mang tính đột phá được giới thiệu vào năm 2017, mở ra một kỷ nguyên mới trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP). Khác với các mô hình truyền thống như mạng nơ-ron tuần hoàn (RNN) hay mạng nơ-ron tích chập (CNN), Transformer sử dụng cơ chế chú ý (attention) tiên tiến, giúp mô hình "hiểu" được mối quan hệ giữa các từ trong một câu một cách sâu sắc và chính xác hơn.

3.2. Cấu trúc
 <img width="497" height="685" alt="image" src="https://github.com/user-attachments/assets/9857c0d5-654c-4faa-9c27-d7d08886fe20" />
Hình 3.1. Cấu trúc mô hình

3.2.1. Bộ mã hóa (Encoder)
-	Chịu trách nhiệm phân tích và biểu diễn chuỗi đầu vào (ví dụ: một câu tiếng Anh).
-	Sử dụng nhiều lớp "encoder" được lặp lại, mỗi lớp gồm hai thành phần chính:
•	Tự chú ý (Self-attention): Cho phép mô hình "tập trung" vào các từ quan trọng trong câu và hiểu được mối quan hệ giữa chúng.
•	Mạng nơ-ron đa tầng (Feed-forward network): Áp dụng các phép toán phi tuyến tính để biến đổi thông tin.
3.2.2. Bộ giải mã (Decoder)
-	Dựa trên thông tin từ bộ mã hóa, tạo ra chuỗi đầu ra (ví dụ: bản dịch tiếng Việt).
-	Tương tự bộ mã hóa, sử dụng nhiều lớp "decoder" được lặp lại, mỗi lớp gồm:
•	Tự chú ý (Self-attention): Xử lý chuỗi đầu ra đang được tạo và mối quan hệ giữa các từ trong nó.
•	Chú ý bộ mã hóa-bộ giải mã (Encoder-decoder attention): Cho phép bộ giải mã "tham khảo" thông tin từ bộ mã hóa để tạo ra bản dịch chính xác hơn.
•	Mạng nơ-ron đa tầng (Feed-forward network): Biến đổi thông tin.

3.3. Encoder
3.3.1. Input Embending
-	Máy tính không hiểu câu chữ mà chỉ đọc được số, vector, ma trận; vì vậy ta phải biểu diễn câu chữ dưới dạng vector, gọi là input embedding. Điều này đảm bảo các từ gần nghĩa có vector gần giống nhau. Hiện đã có khá nhiều pretrained word embeddings như GloVe, Fasttext, gensim Word2Vec,... cho bạn lựa chọn.
 <img width="667" height="281" alt="image" src="https://github.com/user-attachments/assets/973e23be-91ec-4239-ac2d-9b6910f49756" />
Hình 3.2. Biểu diễn từ dưới dạng vector
3.3.2. Positional Encoding
-	Word embeddings phần nào cho giúp ta biểu diễn ngữ nghĩa của một từ, tuy nhiên cùng một từ ở vị trí khác nhau của câu lại mang ý nghĩa khác nhau. Đó là lý do Transformers có thêm một phần Positional Encoding để inject thêm thông tin về vị trí của một từ.
 <img width="527" height="156" alt="image" src="https://github.com/user-attachments/assets/3b6df96c-c518-44bc-8949-87d060e38aa2" />
Hình 3.2. Công thức tính cho các vị trí chẵn/lẽ
Trong đó pos là vị trí của từ trong câu, PE là giá trị phần tử thứ i trong embenddings có độ dài dmodel. Sau đó ta cộng PE Vector và Embendding vector
 <img width="663" height="296" alt="image" src="https://github.com/user-attachments/assets/5f640f1c-f5f2-4b55-9c1f-a36b7af7ff84" />
Hình 3.3. Vector Embbedding
3.3.3. Self-Attention
-	Self-Attention là cơ chế giúp Transformers "hiểu" được sự liên quan giữa các từ trong một câu.
-	Ví dụ như từ "kicked" trong câu "I kicked the ball" (tôi đã đá quả bóng) liên quan như thế nào đến các từ khác? Rõ ràng nó liên quan mật thiết đến từ "I" (chủ ngữ), "kicked" là chính nó lên sẽ luôn "liên quan mạnh" và "ball" (vị ngữ). Ngoài ra từ "the" là giới từ nên sự liên kết với từ "kicked" gần như không có. Vậy Self-Attention trích xuất những sự "liên quan" này như thế nào?
 <img width="607" height="396" alt="image" src="https://github.com/user-attachments/assets/4d8eeaf6-1e05-4670-bae4-5d08a57f614b" />
Hình 3.4. Self-Attention với một từ trong câu
-	Đầu vào của các module Multi-head Attention (bản chất là Self-Attention) có 3 mũi tên, đó chính là 3 vectors Querys (Q), Keys (K) và Values (V). Từ 3 vectors này, ta sẽ tính vector attention Z cho một từ theo công thức sau:
 <img width="689" height="124" alt="image" src="https://github.com/user-attachments/assets/13706796-0487-41e2-a1e4-3faa4100faa7" />
Hình 3.5. Công thức tính Softmax
-	Đầu tiên, để có được 3 vectors Q, K, V, input embeddings được nhân với 3 ma trận trọng số tương ứng (được tune trong quá trình huấn luyện) WQ, WK, WV.
 <img width="671" height="409" alt="image" src="https://github.com/user-attachments/assets/514388cd-b28f-4681-beaa-5e64cadc7bbe" />
Hình 3.6. Vector Q, K, V
-	Lúc này, vector K đóng vai trò như một khóa đại diện cho từ, và Q sẽ truy vấn đến các vector K của các từ trong câu bằng cách nhân chập với những vector này. Mục đích của phép nhân chập để tính toán độ liên quan giữa các từ với nhau. Theo đó, 2 từ liên quan đến nhau sẽ có "Score" lớn và ngược lại.
-	Bước thứ 2 là bước "Scale", đơn giản chỉ là chia "Score" cho căn bậc hai của số chiều của Q/K/V (trong hình chia 8 vì Q/K/V là 64-D vectors). Việc này giúp cho giá trị "Score" không phụ thuộc vào độ dài của vector Q/K/V
-	Bước thứ 3 là softmax các kết quả vừa rồi để đạt được một phân bố xác suất trên các từ.
-	Bước thứ 4 ta nhân phân bố xác suất đó với vector V để loại bỏ những từ không cần thiết (xác suất nhỏ) và giữ lại những từ quan trọng (xác suất lớn).
-	Ở bước cuối cùng, các vectors V (đã được nhân với softmax output) cộng lại với nhau, tạo ra vector attention Z cho một từ. Lặp lại quá trình trên cho tất cả các từ ta được ma trận attention cho 1 câu.
 <img width="637" height="649" alt="image" src="https://github.com/user-attachments/assets/c845648e-3715-4813-be0c-19015a9db150" />
Hình 3.7. Vector Z
 
3.3.4. Multi-Head Attention
-	Vấn đề của Self-attention là attention của một từ sẽ luôn "chú ý" vào chính nó.
 <img width="586" height="300" alt="image" src="https://github.com/user-attachments/assets/675e3552-a0cc-4ee9-909b-e01362af9eb7" />
Hình 3..8 Self-Attetion đối với mỗi từ.
-	Tác giả đã giới thiệu một phiên bản nâng cấp hơn của Self-attention là Multi-head attention. Ý tưởng rất đơn giản là thay vì sử dụng 1 Self-attention (1 head) thì ta sử dụng nhiều Attention khác nhau (multi-head).
-	Vì mỗi "head" sẽ cho ra một ma trận attention riêng nên ta phải concat các ma trận này và nhân với ma trận trọng số WO để ra một ma trận attention duy nhất (weighted sum). Và tất nhiên, ma trận trọng số này cũng được tune trong khi training.
 <img width="784" height="421" alt="image" src="https://github.com/user-attachments/assets/f5bb448e-053f-4eaa-9023-554d0993beb5" />
Hình 3.9. Chuyển đổi về cùng một kích cỡ.
3.3.5. Residuals
-	Mỗi sub-layer đều là một residual block. Cũng giống như residual blocks trong Computer Vision, skip connections trong Transformers cho phép thông tin đi qua sub-layer trực tiếp. Thông tin này (x) được cộng với attention (z) của nó và thực hiện Layer Normalization.
 <img width="486" height="464" alt="image" src="https://github.com/user-attachments/assets/d2f38573-94ff-4204-87cb-84355bc330cf" />
Hình 3.10 Minh họa hướng di chuyển của Residuals
3.3.6. Feed Forward
-	Sau khi được Normalize, các vectors  z được đưa qua mạng fully connected trước khi đẩy qua Decoder. Vì các vectors này không phụ thuộc vào nhau nên ta có thể tận dụng được tính toán song song cho cả câu.
 <img width="736" height="234" alt="image" src="https://github.com/user-attachments/assets/b63c2238-db82-4d9f-810c-d408c362d277" />
Hình 3.11. Minh họa Feed Forward
3.4. Decoder
3.4.1. Masked Multi-Head Attetion
-	Giả sử bạn muốn Transformers thực hiện bài toán English-France translation, thì công việc của Decoder là giải mã thông tin từ Encoder và sinh ra từng từ tiếng Pháp dựa trên NHỮNG TỪ TRƯỚC ĐÓ. Vậy nên, nếu ta sử dụng Multi-head attention trên cả câu như ở Encoder, Decoder sẽ "thấy" luôn từ tiếp theo mà nó cần dịch. Để ngăn điều đó, khi Decoder dịch đến từ thứ i, phần sau của câu tiếng Pháp sẽ bị che lại (masked) và Decoder chỉ được phép "nhìn" thấy phần nó đã dịch trước đó.
 <img width="717" height="325" alt="image" src="https://github.com/user-attachments/assets/a4effbe1-6209-4f39-bb93-95c8f1176184" />
Hình 3.12. Mask-Attention trong giai đoạn giải mã

3.4.2. Quá trình decoder
-	Quá trình decode về cơ bản là giống với encode, chỉ khác là Decoder decode từng từ một và input của Decoder (câu tiếng Pháp) bị masked. Sau khi masked input đưa qua sub-layer #1 của Decoder, nó sẽ không nhân với 3 ma trận trọng số để tạo ra Q, K, V nữa mà chỉ nhân với 1 ma trận trọng số WQ. K và V được lấy từ Encoder cùng với Q từ Masked multi-head attention đưa vào sub-layer #2 và #3 tương tự như Encoder. Cuối cùng, các vector được đẩy vào lớp Linear (là 1 mạng Fully Connected) theo sau bới Softmax để cho ra xác suất của từ tiếp theo.
